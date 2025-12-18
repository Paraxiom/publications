# Closure Checklist

Execute in order. Each step takes < 5 minutes.

---

## Step 1: Upload Primary Paper to Zenodo

1. Go to https://zenodo.org/uploads/new
2. Upload: `paper/paper-core.pdf`
3. Copy metadata from `ZENODO-PRIMARY.md`
4. Publish → get DOI
5. Record DOI here: `__________________`

---

## Step 2: Upload Appendix to Zenodo

1. Go to https://zenodo.org/uploads/new
2. Upload:
   - `paper/paper-appendix.pdf`
   - `paper/beamer-deck.pdf`
   - `diagrams/coherence-pipeline.pdf`
   - `diagrams/control-loop.pdf`
   - `diagrams/threat-defense.pdf`
3. Copy metadata from `ZENODO-APPENDIX.md`
4. Add Related Identifier: "Supplements" → primary DOI from Step 1
5. Publish → get DOI
6. Record DOI here: `__________________`

---

## Step 3: Link Primary to Appendix

1. Go to primary publication on Zenodo
2. Edit metadata
3. Add Related Identifier: "Is supplemented by" → appendix DOI from Step 2
4. Save

---

## Step 4: Update README with DOIs

1. Edit `README.md`
2. Replace `[pending]` with actual DOIs
3. Update citation block

---

## Step 5: Create GitHub Repository

Option A: New repo
```bash
cd /Users/sylvaincormier/paraxiom/publications/papers/augmented-democracy
git init
git add README.md paper/ diagrams/
git commit -m "Augmented Democracy: Coherence-Constrained Control System (archive)"
gh repo create augmented-democracy-coherence --public --source=. --push
```

Option B: Push to existing repo
```bash
# adjust as needed
```

---

## Step 6: Announce (once, quietly)

Optional. One post, one place. Suggested text:

> Published: "Augmented Democracy as a Coherence-Constrained Control System"
>
> A procedural governance framework treating legitimacy as process quality under adversarial conditions.
>
> Paper: [Zenodo DOI]
> Archive: [GitHub URL]
>
> No product. No token. Just infrastructure logic.

---

## Step 7: Walk Away

- Close this directory
- Do not revisit for 6 months
- Work on something that pays

---

## Completed

- [ ] Step 1: Primary Zenodo
- [ ] Step 2: Appendix Zenodo
- [ ] Step 3: Link publications
- [ ] Step 4: Update README
- [ ] Step 5: GitHub archive
- [ ] Step 6: Announce
- [ ] Step 7: Disengage
