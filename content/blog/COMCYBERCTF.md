+++
title = 'COMCYBER CTF'
date = 2026-08-09T12:34:36-04:00
description = "Cet article se base sur la résolution d'un challenge de cyber sécurité développé par le Commandement de la Cyber Défense (COMCYBER)"
tags = ["CTF", "COMCYBER", "OPSEC"]
categories = ["CTF Writeup"]
+++


WriteUp : By Youness LAGNAOUI

Cet article se base sur la résolution du challenge développé par le Commandement de la Cyber Defense (COMCYBER) dans le cadre des campagnes de recrutement. Le challenge est accessible via Root Me à l'adresse : https://www.root-me.org/fr/Challenges/Realiste/ComCyber-Challenge

![Resize](/images/comcyber/0.png?width=200px)



## Mission 1 : Post-mortem communication véhicule


![Resize](/images/comcyber/1.png?width=200px)


![Resize](/images/comcyber/2.png?width=200px)

![Resize](/images/comcyber/3.png?width=200px)

Nous avons à notre disposition une capture réseau correspondant à une communication via protocole TELNET avec le véhicule.

TELNET est un protocole non chiffré nous permettant de lire directement le contenu des trames échangées !

Effectuons un focus sur les dernières trames pour identifier ce qui a causé la perte de contact avec le véhicule :

![Resize](/images/comcyber/4.png?width=200px)


On observe que l'intégralité des communications TELNET peut être lue, nous permettant d'identifier un couple username / mot de passe :

```
root:vlR0IQwJL60RGQvA8ne2mopXZk
```

![Resize](/images/comcyber/5.png?width=200px)

Finalement, on n'observe pas grand-chose de si intéressant au sein des communications TELNET qui permettrait d'expliquer la perte de connexion au véhicule.

Les connexions TELNET sont un rabbit hole.

Analysons les autres types de communication que nous avons : les communications UDP :

En prenant un paquet UDP aléatoire, on obtient :


![Resize](/images/comcyber/6.png?width=200px)


Analysons le dernier paquet UDP transmis au véhicule :

![Resize](/images/comcyber/7.png?width=200px)

Cette trame UDP nous donne l'indication de nous connecter à l'adresse IP :

```
212.129.38.224:53036
```


## Mission 2 : Accéder au tableau de bord

![Resize](/images/comcyber/8.png?width=200px)

On tombe sur une mire d'authentification.


Essayons les techniques de bypass de login classiques de type injection SQL pour voir si ça passe :

```
Username = ' OR 1=1; --
```


![Resize](/images/comcyber/9.png?width=200px)


![Resize](/images/comcyber/10.png?width=200px)


On a bien contourné le mécanisme d'authentification via injection SQL. On accède à un dashboard permettant de lister les véhicules déployés. Parmi cette liste, un véhicule apparaît comme compromis et inaccessible : le véhicule **GOLIATH** :

![Resize](/images/comcyber/11.png?width=200px)


Nous avons 2 options de contrôle réalisables :

- *Enable Comunications*
- *Get last logs*

Quand on clique sur *Enable Comunications*, on obtient cette fenêtre demandant un mot de passe d'activation :

![Resize](/images/comcyber/12.png?width=200px)

Testons le mot de passe que nous avons pu identifier dans les échanges TELNET de la première étape du challenge :

![Resize](/images/comcyber/12_2.png?width=200px)

Code non validé...

Lorsque nous cliquons sur "Get last logs", on obtient cette fenêtre :

![Resize](/images/comcyber/13.png?width=200px)

qui nous permet de télécharger ce fichier .log contenant des logs système d'une machine Linux :

![Resize](/images/comcyber/14.png?width=200px)

Cherchons les commandes système effectuées sur cette machine en filtrant sur le préfixe **CMD** :

![Resize](/images/comcyber/15.png?width=200px)

On observe la présence d'une commande visant à stocker la chaîne de caractères ```efa3fb1f409aa622dddd7220ad6567580fb68b68``` en tant que potentiel password.

On observe que l'attaquant a téléchargé sur le serveur COMCYBER un fichier nommé **m4lw3r3** depuis son serveur (```attacker[.]tld```) et l'a zippé avec le mot de passe ```efa3fb1f409aa622dddd7220ad6567580fb68b68```.

![Resize](/images/comcyber/16.png?width=200px)

On observe également que le fichier Zip contenant le fameux fichier **m4lw3r3** se situe au sein de l'image favicon.ico du site :

![Resize](/images/comcyber/17.png?width=200px)


Téléchargeons le fichier favicon.ico afin d'observer ce qu'il y a dedans :

![Resize](/images/comcyber/18.png?width=200px)


![Resize](/images/comcyber/19.png?width=200px)


En inspectant le contenu des deux fichiers obtenus grâce à l'utilitaire binwalk, on observe que le fichier **favicon.ico.1** contient une archive zip contenant les 2 fichiers :

- message.txt
- m4lw4r3

Effectuons une extraction de ces fichiers grâce à binwalk :

![Resize](/images/comcyber/20.png?width=200px)

![Resize](/images/comcyber/21.png?width=200px)


Les fichiers ne sont pas directement accessibles car ils n'ont pas été extraits depuis le zip protégé par mot de passe.

Faisons une extraction du zip avec le mot de passe identifié dans les logs : ```efa3fb1f409aa622dddd7220ad6567580fb68b68``` :

![Resize](/images/comcyber/22.png?width=200px)

Maintenant nous avons bien accès à notre fameux fichier **m4lw3r3** qui nous intéresse.


## Mission 3 : Reverse Engineering Malware

Nous avons maintenant à notre disposition le malware qui a été responsable de la perte de connexion avec le véhicule **GOLIATH**. Analysons-le :


![Resize](/images/comcyber/23.png?width=200px)


On peut constater que c'est un binaire Linux non strippé, ce qui va faciliter l'analyse statique du binaire.


![Resize](/images/comcyber/24.png?width=200px)

Dans les chaînes de caractères du binaire, on peut observer une chaîne chiffrée (peut-être la chaîne de caractères à déchiffrer pour finir le challenge ?)

Utilisons Ghidra pour analyser plus en détail le binaire :

Le binaire prend en entrée 2 inputs :

![Resize](/images/comcyber/24_2.png?width=200px)

![Resize](/images/comcyber/25.png?width=200px)


Le binaire fait une vérification de signature **RSA brute (RSA_NO_PADDING)** avec une clé publique `(n, e)` embarquée.

1. **input1** doit être égal à une chaîne fixe stockée dans la table `strs` (indice 77) :
    `6762f71489bb7b57bbdaabd1306a677bbcf25bd7` → sinon `input1 FAILED`.
2. Cette chaîne est répétée par `strncat` jusqu'à **128 octets** → c'est le message `tbs`.
3. **input2** est décodé depuis l'hex → c'est la **signature**. Le binaire fait `EVP_PKEY_verify` avec padding `3` (= `RSA_NO_PADDING`), donc il teste simplement `sig^e mod n == tbs`.
4. Si les deux passent → il affiche `You should be able to decrypt this: 0x39c4…` (un chiffré 1024 bits).

`e` fait **1024 bits** (presque aussi grand que `n`). On tombe dans un cas classique de clé vulnérable aux attaques de **Wiener** (fractions continues de `e/n`).

Demandons à Claude de nous faire les calculs pour retrouver le secret final :

![Resize](/images/comcyber/26.png?width=200px)

![Resize](/images/comcyber/27.png?width=200px)

On obtient la chaîne de caractères :


```
3302758350382e964f2931b596c5abe2
```

Vérifions que cela permet de rétablir le contact avec le véhicule **GOLIATH** :

![Resize](/images/comcyber/28.png?width=200px)


Communication avec le véhicule **GOLIATH** rétablie.

Flag final :


```
RM{F33ling_L1k3_4_R34l_Ag3nT!}
```