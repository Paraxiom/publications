# Preuve de Cohérence

**Une couche de gouvernance et de vérification pour les maillages IA fédérés**

---

## Énoncé du problème

Les systèmes d'IA fédérés et distribués reposent sur des nœuds non fiables et hétérogènes qui contribuent :
- des mises à jour de modèles,
- des sorties d'inférence,
- ou du travail d'entraînement local.

Aujourd'hui, les critères d'acceptation sont faibles :
- agrégation statistique,
- confiance envers les opérateurs,
- ou détection grossière d'anomalies.

Cela crée quatre risques :
1. **Contributions hallucinées ou incohérentes** acceptées dans les modèles partagés
2. **Déstabilisation silencieuse** des modèles globaux par des nœuds malveillants
3. **Aucune notion objective de « travail utile »** dans les maillages IA
4. **Gaspillage d'énergie** sur la vérification redondante par recalcul

Il n'existe actuellement aucun équivalent de « preuve de travail utile » pour l'IA distribuée.

---

## Idée centrale

La Preuve de Cohérence est une couche de vérification légère qui prouve qu'une contribution IA était :
- cohérente en interne,
- non divergente,
- et sûre à accepter dans un système IA partagé,

**sans faire confiance au nœud** qui l'a produite et **sans réexécuter le modèle**.

> Nous ne prouvons pas qu'une IA a raison — nous prouvons qu'elle est restée cohérente.

---

## Ce que « Cohérence » signifie (opérationnellement)

Chaque contribution IA produit, en plus de sa sortie, une **empreinte de cohérence** composée de :

| Signal | Ce qu'il mesure | Calcul |
|--------|-----------------|--------|
| **Entropie d'attention** | Stabilité de distribution entre les têtes | O(n) par couche |
| **CV spectral** | Coefficient de variation des valeurs propres d'attention | O(n²) une fois |
| **Consistance topologique** | Déviation des voisinages d'attention attendus | O(n) par couche |
| **Taux de dérive** | Changement des normes d'état caché entre couches | O(1) par couche |

Ces signaux sont :
- **Conscients de l'architecture** — calibrés par famille de modèle
- **Peu coûteux à calculer** — ~5% de surcharge pendant l'inférence
- **Déterministes à vérifier** — même entrée → même empreinte

Ils répondent à une seule question :

> Cette contribution IA est-elle restée dans les limites de cohérence acceptables ?

---

## Intégration dans l'apprentissage fédéré

**Flux existant :**
```
Nœud → Entraînement / Inférence local → Mise à jour envoyée → Agrégation
```

**Avec Preuve de Cohérence :**
```
Nœud → Entraînement / Inférence local
     → Empreinte de cohérence (5% surcharge)
     → Porte de vérification de cohérence (0,1% surcharge)
     → Accepté ou Rejeté
     → Agrégation
```

**Aucun changement aux :**
- algorithmes d'entraînement,
- architectures de modèles,
- ou outils d'orchestration.

C'est une **couche de gouvernance parallèle**, pas une réécriture.

---

## Pourquoi c'est important

### 1. Sensible à l'architecture
Différents modèles ont différentes enveloppes de stabilité. Notre recherche le démontre empiriquement :

| Modèle | Effet de la contrainte toroïdale |
|--------|----------------------------------|
| Phi-2 (2,78B) | 50% de **réduction** des hallucinations |
| TinyLlama (1,1B) | 180% d'**augmentation** des hallucinations |

*Même contrainte, effet opposé.* Les corrections universelles n'existent pas. La gouvernance doit être consciente de l'architecture.

### 2. Conscient des hallucinations
Les hallucinations sont traitées comme des **échecs de cohérence**, pas seulement des erreurs factuelles. Un modèle qui dérive hors de son enveloppe de cohérence est signalé avant que sa sortie ne se propage.

### 3. Résistant aux adversaires
Les empreintes de cohérence sont dérivées des dynamiques internes du modèle (patterns d'attention, évolution de l'état caché). Falsifier une empreinte valide nécessite :
- Accès aux poids du modèle
- Production de sorties qui suivent réellement des chemins d'attention cohérents
- Correspondance des signatures spectrales

Falsifier la cohérence est computationnellement équivalent à faire un travail cohérent.

### 4. Économe en énergie
La vérification traditionnelle nécessite un **recalcul** — réexécuter le même modèle pour vérifier les résultats.

La Preuve de Cohérence ne nécessite que la **vérification d'empreinte** :

| Approche | Coût énergétique |
|----------|------------------|
| Recalcul complet | 100% |
| Vérification Preuve de Cohérence | **< 1%** |

Pour l'IA fédérée à grande échelle, cela se traduit par :
- **Réduction de 99%+** de l'énergie de vérification
- Aucun cycle GPU redondant
- Empreinte carbone proportionnelle au travail utile, pas à la paranoïa

> C'est une gouvernance IA verte : la confiance par les mathématiques, pas par la force brute.

---

## Preuve de travail utile (reformulée)

Les systèmes traditionnels prouvent :
- l'énergie brûlée (PoW),
- ou le capital verrouillé (PoS).

La Preuve de Cohérence prouve :

> **Ce travail IA était utile car il a préservé ou amélioré la stabilité du système.**

Cela crée une notion mesurable et applicable de **contribution IA utile** dans un maillage distribué — sans gaspiller d'énergie sur la vérification redondante.

---

## Audit et attestation optionnels

Si requis :
- Les attestations de cohérence peuvent être **enregistrées de manière immuable**
- Permet l'auditabilité, la responsabilité et la confiance inter-organisations
- Compatible avec les cadres de conformité existants

Cette couche est optionnelle et n'affecte pas le fonctionnement IA de base.

---

## Statut actuel

**Implémenté et testé sur de vrais LLMs :**
- Démontré que les contraintes de cohérence affectent les taux d'hallucination
- Découverte d'effets dépendants de l'architecture (critique pour la conception de gouvernance)
- Résultats de divergence publiés : [DOI: 10.5281/zenodo.18267913](https://doi.org/10.5281/zenodo.18267913)
- Code ouvert : [github.com/Paraxiom/topological-coherence](https://github.com/Paraxiom/topological-coherence)

**R&D active :**
- Calibration consciente de l'architecture (en cours)
- Algorithmes de sélection de topologie
- Optimisation des hyperparamètres pour différentes familles de modèles

---

## Ce que cela permet pour Numana

| Capacité | Bénéfice |
|----------|----------|
| **Primitive de gouvernance** | Accepter/rejeter les contributions IA objectivement |
| **Confiance sans contrôle central** | Les nœuds prouvent leur propre cohérence |
| **Efficacité énergétique** | Vérifier sans recalculer |
| **Auditabilité** | Enregistrements de cohérence immuables |
| **Scalabilité** | Vérification O(1) par contribution |

---

## Résumé en une phrase

> **La Preuve de Cohérence est une couche de vérification qui prouve que le travail IA distribué était stable et utile — pas seulement calculé — avant qu'il ne soit accepté par le système, à 99% moins de coût énergétique que le recalcul.**

---

## Contact

**Sylvain Cormier**
Paraxiom Research
@ParaxiomAPI
research@paraxiom.io

---

*Recherche appuyée par des résultats publiés. Code disponible pour revue technique.*
