# INDUSTRIAL PROCESS MONITORING - IMPLEMENTATION TODO

## ✅ STEP 1: Update requirements.txt
- Added scipy, seaborn for hierarchical clustering

## ✅ STEP 2: Update pre_process.py  
- Added time sorting by index (no explicit time column)
- Added lag1/lag2 features, rolling mean/std (window=3)
- Data now sorted chronologically, no shuffle

## [ ] STEP 3: Update main.py - CHANGE 1 (Problem Definition)
- Replace build_problem_payload() with EXACT spec content

## ✅ STEP 4: main.py - CHANGE 2 (Predictive Models)
- ✅ Uses lag/rolling features from preprocess
- ✅ Chronological split (no shuffle)
- ✅ Full metrics: accuracy/precision/recall/F1/ROC-AUC
- ✅ LogisticRegression fixed (max_iter=2000)

## ✅ STEP 5: main.py - CHANGE 3 (Clustering)
- ✅ Added hierarchical clustering + dendrogram data
- ✅ KMeans preserved + new hierarchical_labels field

## [ ] STEP 6: Dataset Persistence
- Already working via global current_data ✅

## ✅ STEP 7: index.html - CHANGE 5 (UI Overflow)
- ✅ Fixed: h-screen → min-h-screen
- ✅ Added max-h-[calc(100vh-12rem)] to main content
- ✅ Updated Problem Definition to match spec text

## [ ] STEP 8: Testing & Install
- pip install -r requirements.txt
- uvicorn main:app --reload
- Test all endpoints + persistence + UI

