# Comprendre la compilation avec CMake

Ce guide part d'un petit programme C++ et aboutit aux générateurs du projet.
Pour la liste des modules CMake et leurs responsabilités, voir ensuite
[l'organisation des cibles](../cmake/README.md).

## Le point de départ : compiler un petit programme

Supposons trois fichiers :

```cpp
// calcul.hpp
int doubler(int nombre);
```

```cpp
// calcul.cpp
#include "calcul.hpp"

int doubler(int nombre) { return 2 * nombre; }
```

```cpp
// main.cpp
#include "calcul.hpp"
#include <iostream>

int main() {
    std::cout << doubler(21) << '\n';
}
```

On peut produire un exécutable sans CMake :

```bash
g++ -std=c++20 main.cpp calcul.cpp -o calcul
./calcul
```

`main.cpp` contient le point d'entrée `main`. `calcul.hpp` déclare la fonction
pour que les deux fichiers C++ connaissent sa signature. Le compilateur compile
`main.cpp` et `calcul.cpp` séparément ; l'éditeur de liens réunit ensuite leurs
résultats. Il ne devine pas que `calcul.cpp` accompagne `calcul.hpp` parce que
leurs noms se ressemblent : il faut lui donner le fichier à compiler. Si on
omet `calcul.cpp`, la définition de `doubler` manque au moment du lien. Si deux
fichiers fournissent chacun un `main`, le lien échoue également.

`-o calcul` choisit le **nom du fichier produit**. Cela ne désigne ni le fichier
source principal ni une règle de compilation. Dans cet exemple, l'exécutable
est `./calcul`.

## Ce que CMake ajoute

Dans un projet qui possède plusieurs exécutables et bibliothèques, écrire à
la main une grande commande `g++` pour chacun devient pénible. On décrit
plutôt les cibles dans un fichier `CMakeLists.txt` :

```cmake
cmake_minimum_required(VERSION 3.20)
project(exemple LANGUAGES CXX)
add_executable(calcul main.cpp calcul.cpp)
target_compile_features(calcul PRIVATE cxx_std_20)
```

`add_executable(calcul main.cpp calcul.cpp)` dit : « crée une cible nommée
`calcul` à partir de ces deux sources ». C'est le premier argument, `calcul`,
qui donne son nom à la cible et, ici, à l'exécutable. Ni `main.cpp` ni
`calcul.cpp` ne prime pour choisir ce nom. Une cible peut aussi être une
bibliothèque (`add_library`) ou un groupe de cibles (`add_custom_target`) :
toutes les cibles ne produisent donc pas un exécutable.

Deux commandes suffisent alors :

```bash
cmake -S . -B build -G Ninja
cmake --build build --target calcul
./build/calcul
```

La première commande **configure** le projet. `-S .` indique le dossier des
sources, ici le dossier courant : CMake y lit `CMakeLists.txt`, qui peut à son
tour inclure d'autres fichiers CMake. `-B build` choisit le dossier de travail
de la compilation ; CMake le crée s'il n'existe pas. `-G Ninja` lui demande
d'écrire les règles destinées à Ninja. Cette étape ne produit pas encore
`build/calcul`.

La deuxième commande **construit** la cible `calcul`. CMake appelle Ninja avec
les règles qu'il vient de préparer ; Ninja compile les sources nécessaires,
lie leurs résultats et écrit `build/calcul`. La troisième commande exécute le
programme. CMake ne l'exécute pas à notre place.

Dans cet exemple, le dossier `build/` contient notamment `CMakeCache.txt` (les
options de configuration retenues), `build.ninja` (les règles remises à Ninja),
`CMakeFiles/` (des fichiers intermédiaires), des bibliothèques et les
exécutables effectivement construits. On n'y écrit pas les sources à la main.

## Notre configuration `ppti`

Le fichier [CMakePresets.json](../CMakePresets.json) enregistre une
configuration nommée `ppti`. Elle cible CUDA 12.9 et les GPU V100 (SM70),
utilise C++20 et GNU Make, active les tests et place les résultats dans
`builds/ppti-gpu4/`. Le paquet mathDx 25.12.1 CUDA 12 et un compilateur
hôte compatible avec CUDA 12.9 doivent être disponibles avant la configuration.

Depuis la racine du projet, lancer la configuration :

```bash
cmake --preset ppti
```

Si les chemins locaux diffèrent de ceux du preset, fournir
`CMAKE_CUDA_COMPILER`, `CMAKE_CUDA_HOST_COMPILER`, `CMAKE_CXX_COMPILER` et
`AI_FACTORY_MATHDX_ROOT` avec `-D`. CMake lit `CMakeLists.txt` et les modules
sous `cmake/` ; les sources et les règles ne viennent jamais du dossier de build.

`-DVARIABLE=VALEUR` sert à définir ou remplacer une option de configuration.
`-j2`, utilisé avec `cmake --build`, autorise deux tâches de compilation en
parallèle ; cela ne règle pas le nombre de chemins Monte Carlo ou les threads
GPU d'une simulation.

Les *presets de build* choisissent des cibles. Par exemple,
`cmake --build --preset host-tests` construit les tests hôte après la
configuration du projet. Sur la RTX 4090 locale, `cmake --preset local-sm89`
utilise le même socle C++20/CUDA 12.9 et crée
`builds/local-cuda12-9-0-sm89/` ; le choix de l'architecture et le répertoire
de build restent séparés de ceux du serveur PPTI. Sur le répertoire personnel
partagé de PPTI, `cmake --preset ppti-gpu1` et `cmake --preset ppti-gpu3`
utilisent CUDA 13.2/SM120 et créent respectivement `builds/ppti-gpu1/` et
`builds/ppti-gpu3/`. Le preset `ppti` crée `builds/ppti-gpu4/` en CUDA
12.9/SM70. Chaque nœud compile ses propres objets et exécutables ; le transfert
des sources ne copie aucun dossier `builds/`.

## Construire un générateur précis

Le [module CMake du catalogue](../cmake/AIFactoryCatalog.cmake) enregistre les
générateurs avec `add_executable` et relie les bibliothèques nécessaires avec
`target_link_libraries`. Les noms des recettes et des cibles sont fournis par
les manifestes générés du projet. Un `#include` dans un `.cpp` rend les
déclarations disponibles au compilateur ; il ne remplace ni la compilation des
implémentations ni leur liaison à l'exécutable. Pour une cible de prix, la
fonction CMake `add_price_generator` détermine les unités de pricing requises,
les relie à `ai_factory_price_dataset` et enregistre le `generator.cpp` comme
source de l'exécutable. Quand on construit cette cible, Make construit aussi
les bibliothèques liées qui manquent, puis fait le lien final. Il n'est pas
nécessaire de nommer chaque `.cu` et `.cpp` dans la commande `cmake --build`.

Après la configuration, on peut construire exactement le générateur Heston
cartésien prix + gradient + Hessienne diagonale :

```bash
cmake --build builds/ppti-gpu4 --target generate_heston_european_calls_01_cartesian_price_gradient_diagonal_hessian -j2
```

Cette cible produit
`builds/ppti-gpu4/generate_heston_european_calls_01_cartesian_price_gradient_diagonal_hessian`
et correspond au fichier source
`work/catalog/model/equity/markovian/heston/price_gradients/european_calls/heston_01__european_calls_01__01_cartesian_price_gradient_diagonal_hessian/generator.cpp`.
Cette recette locale se trouve sous `work/` et n'est pas versionnée.
Le nom de la cible n'est donc pas `generator.cpp` : ce nom de fichier est
réutilisé par de nombreuses recettes.

`--target` attend un **nom de cible connu de CMake**. Cette commande montre
les cibles principales :

```bash
cmake --build builds/ppti-gpu4 --target help
```

Dans ce grand catalogue, elle ne montre pas tous les générateurs individuels.
Pour chercher les cibles Heston dans ce build, afficher les cibles disponibles :

```bash
cmake --build builds/ppti-gpu4 --target help | rg 'generate_heston_'
```

On peut demander une cible de regroupement, comme `price_gradient_generators`,
mais cela construit beaucoup plus de générateurs. Les générateurs individuels
sont exclus de la construction par défaut (`EXCLUDE_FROM_ALL`) : choisir leur
cible évite de compiler tout le catalogue.

**Compiler n'est pas générer une base.** Les commandes `cmake --build`
construisent des exécutables ; elles ne les lancent pas. Une commande distincte
comme `./builds/ppti-gpu4/generate_heston_european_calls_01_cartesian_price_gradient_diagonal_hessian`
déclenche la simulation et écrit des données. Pour préparer, suivre, publier ou
reprendre une campagne, voir le
[workflow de génération](dataset-generation-workflow.md). Le notebook
Heston puis rough Heston sous
`work/experiments/equity/cross_model/heston_rough_heston_price_delta_generation/notebook.ipynb`
construit chaque cible avant de lancer son exécutable.

## Après une modification du code

Si on redemande une cible déjà construite, Make regarde les fichiers modifiés
et leurs dépendances. Il recompile ce qui doit l'être, puis refait les liens
concernés. Si rien n'a changé, aucune ligne `Building` ou `Linking` n'apparaît.
Il n'y a pas
besoin d'effacer `builds/ppti-gpu4/` pour mettre un exécutable à jour.

`builds/ppti-gpu4/` est ignoré par Git : ce sont des résultats locaux, dépendants de la
machine et des outils. GitHub conserve les sources, `CMakeLists.txt`, les
modules `cmake/` et `CMakePresets.json`, qui permettent de recréer ce dossier.
Un second build n'est utile que pour garder en parallèle une autre
configuration, par exemple un autre GPU ou compilateur. On peut alors utiliser
`cmake -S . -B builds/sm_XX -G "Unix Makefiles" ...` : CMake crée ce nouveau dossier et
chaque build garde ses propres exécutables. Les builds sont
`builds/local-cuda12-9-0-sm89/` en local et `builds/ppti-gpu1/`,
`builds/ppti-gpu3/` ou `builds/ppti-gpu4/` sur le nœud PPTI choisi ;
les preuves historiques et campagnes sont rangées séparément selon
le [plan des artefacts locaux](local-artifacts.md).
