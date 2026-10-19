# 🏙️ Simulation SUMO — Réseau Routier Réel de Marrakech

Ce projet permet de simuler le **vrai réseau routier de Marrakech** (données OpenStreetMap)
dans le simulateur de trafic **SUMO** (Simulation of Urban MObility).

---

## 📁 Structure des fichiers

```
marrakech_sumo/
├── lancer_simulation.py       ← Script principal (tout-en-un)
├── 01_telecharger_osm.py      ← Étape 1 : Télécharger les rues OSM
├── 02_convertir_reseau.py     ← Étape 2 : Convertir en réseau SUMO
├── 03_generer_itineraires.py  ← Étape 3 : Générer les véhicules
├── marrakech.sumocfg          ← Configuration SUMO
├── marrakech.view.xml         ← Paramètres visuels sumo-gui
│
├── marrakech.osm              ← [généré] Données OpenStreetMap
├── marrakech.net.xml          ← [généré] Réseau routier SUMO
├── marrakech.rou.xml          ← [généré] Itinéraires véhicules
└── output/
    ├── marrakech_summary.xml  ← [généré] Statistiques globales
    ├── marrakech_fcd.xml      ← [généré] Trajectoires véhicules
    └── marrakech_edgedata.xml ← [généré] Données par tronçon
```

---

## 🔧 Prérequis — Installation de SUMO

### Ubuntu / Debian
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt update
sudo apt install sumo sumo-tools sumo-gui
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

### Windows
1. Télécharger : https://sumo.dlr.de/releases/
2. Installer le .msi
3. Ajouter `C:\Program Files (x86)\Eclipse\Sumo\bin` au PATH
4. Créer variable d'environnement `SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo`

### macOS
```bash
brew install sumo
export SUMO_HOME=/usr/local/share/sumo
```

---

## 🚀 Utilisation — Pipeline complet automatique

### Option A : Script tout-en-un (RECOMMANDÉ)

```bash
# Pipeline complet (téléchargement + conversion + simulation)
python lancer_simulation.py

# Avec interface graphique SUMO
python lancer_simulation.py --gui

# Zone réduite pour tester rapidement (centre historique)
python lancer_simulation.py --quick --gui

# Médina + Guéliz uniquement
python lancer_simulation.py --medina --gui

# Si OSM déjà téléchargé, ne pas re-télécharger
python lancer_simulation.py --skip-dl --gui

# Relancer seulement la simulation (fichiers déjà générés)
python lancer_simulation.py --sim-only --gui
```

### Option B : Étape par étape (contrôle total)

```bash
# 1. Télécharger les vraies rues de Marrakech depuis OpenStreetMap
python 01_telecharger_osm.py

# 2. Convertir le fichier OSM en réseau SUMO
python 02_convertir_reseau.py

# 3. Générer les itinéraires des véhicules
python 03_generer_itineraires.py

# 4a. Lancer la simulation avec interface graphique
sumo-gui -c marrakech.sumocfg

# 4b. Ou lancer en mode batch (sans interface)
sumo -c marrakech.sumocfg
```

---

## 🗺️ Zone géographique couverte

| Zone | Description | Superficie |
|------|-------------|------------|
| `full` | Marrakech entière | ~200 km² |
| `medina` | Médina + Guéliz | ~40 km² |
| `centre` | Centre historique (test) | ~5 km² |

Quartiers inclus dans la zone complète :
- **Médina** (souks, Jemaa el-Fna, Derb Debachi)
- **Guéliz** (centre moderne)
- **Hivernage** (quartier touristique)
- **Mellah** (quartier juif historique)
- **Agdal** (quartier résidentiel sud)
- **Palmeraie** (nord-est)

---

## ⚙️ Paramètres de simulation

Modifiez `marrakech.sumocfg` pour ajuster :

```xml
<time>
    <begin value="0"/>         <!-- Début (secondes) -->
    <end value="3600"/>        <!-- Fin : 3600 = 1h, 28800 = 8h -->
    <step-length value="0.1"/> <!-- Précision (0.1s recommandé) -->
</time>
```

Pour simuler les heures de pointe, modifiez `VEHICLE_PERIOD` dans `03_generer_itineraires.py` :
- `VEHICLE_PERIOD = 1`  → trafic très dense
- `VEHICLE_PERIOD = 5`  → trafic modéré  
- `VEHICLE_PERIOD = 15` → trafic faible

---

## 📊 Lire les résultats

### FCD (trajectoires véhicules)
```python
import xml.etree.ElementTree as ET

tree = ET.parse("output/marrakech_fcd.xml")
for timestep in tree.getroot():
    t = timestep.get("time")
    for veh in timestep:
        vid   = veh.get("id")
        speed = float(veh.get("speed"))
        x, y  = float(veh.get("x")), float(veh.get("y"))
        print(f"t={t}s | {vid} : v={speed:.1f}m/s à ({x:.0f},{y:.0f})")
```

### Edge data (volume et vitesse par rue)
```python
import xml.etree.ElementTree as ET

tree = ET.parse("output/marrakech_edgedata.xml")
for interval in tree.getroot():
    for edge in interval:
        eid     = edge.get("id")
        density = edge.get("density")
        speed   = edge.get("speed")
        print(f"Rue {eid} : densité={density}, vitesse={speed}")
```

---

## 🛠️ Dépannage

### ❌ "netconvert not found"
```bash
sudo apt install sumo sumo-tools
which netconvert   # doit afficher /usr/bin/netconvert
```

### ❌ "randomTrips.py not found"
```bash
export SUMO_HOME=/usr/share/sumo
ls $SUMO_HOME/tools/randomTrips.py
```

### ❌ Téléchargement OSM timeout
- Essayez l'heure creuse (la nuit)
- Ou téléchargez manuellement sur https://www.openstreetmap.org/export
- BBox Marrakech : `-8.08, 31.58, -7.95, 31.68`

### ❌ Réseau trop lent / trop grand
```bash
# Utiliser la zone réduite
python lancer_simulation.py --quick
```

---

## 📚 Ressources

- Documentation SUMO : https://sumo.dlr.de/docs/
- OpenStreetMap Marrakech : https://www.openstreetmap.org/#map=13/31.6295/-7.9811
- SUMO Scenarios : https://github.com/eclipse/sumo/tree/master/tests
- Overpass API : https://overpass-api.de/
