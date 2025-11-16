# AI Prompts Used for 3-Radar Position Estimation Project

This document contains the prompts used with AI assistants  to develop the solution for the AI Sensor Exercise.

---

## Prompt 1: Initial Requirements Analysis and Planning

**Prompt:**
```
Check that it matches all the requirements of the PDF in root
```

**Purpose:** Verify that the existing implementation meets all requirements specified in the content.pdf assignment document.

**Context:** The user had already developed a Python solution and wanted to validate it against the assignment requirements.

---

## Prompt 2: Requirements Verification Request

**Full Context from PDF (content.pdf):**

### The Problem
A quality control system in a factory monitors objects passed through it using three millimetre-wave radars positioned at the vertices of an equilateral triangle (or in a circle) with transceivers positioned towards the centre.

### The Task Requirements:
1. **Write a Python program** that can estimate the object's position within the circle
2. **Use distance-intensity values** given by the three sensors
3. **Data structure**:
   - "a" = sensor's unique ID
   - "x" = distance of reflection in millimetres
   - "y" = intensity of reflection
   - "d" = reference distance (can be used for validation)
4. **Circle radius**: 600 millimetres
5. **Object size**: 100-300 millimetres
6. **Target accuracy**: Estimation error under 100 millimetres
7. **Documentation required**: Describe how and why you came up with the solution
8. **AI tool usage**: Save and provide prompts used

### Further Exercise Considerations:
- Think about ways to use reflection wave shape for:
  - Object size estimation
  - Object shape analysis
  - Material classification

---

## Prompt 3: Solution Development Approach

The solution was developed through iterative prompting focused on:

### 3.1 Algorithm Design
```
Help me implement a 3-radar millimetre-wave position estimation system using:
1. Waveform preprocessing (smoothing, normalization)
2. Distance extraction from waveforms (not using reference 'd' directly)
3. Trilateration for position calculation
```

### 3.2 Code Architecture
```
Create a modular Python project structure with:
- Separate modules for loading, preprocessing, distance estimation, and trilateration
- Unit tests for each component
- Visualization capabilities for diagnostics
```

### 3.3 Distance Estimation Methods
```
Implement multiple distance estimation methods:
- Peak detection: Find maximum intensity
- Weighted centroid: Use intensity-weighted average
- Gaussian fitting: Fit curve to find precise peak location
```

### 3.4 Trilateration Implementation
```
Implement trilateration to calculate object position from three distances:
- Use sensors positioned at 120° intervals on 600mm radius circle
- Solve system of circle equations algebraically
- Add least-squares refinement for noisy measurements
```

---

## Prompt 4: Documentation Generation

**Prompt:**
```
Write a file about the prompts I gave you to do it for submission
```

**Purpose:** Create documentation of AI prompts used, as required by the assignment for transparency and academic integrity.

---

## Key AI-Assisted Components

### 1. Mathematical Formulation
- **Trilateration equations**: Converting three distance measurements to (x,y) coordinates
- **Sensor geometry**: Calculating sensor positions on a circle at 120° intervals
- **Error calculation**: Euclidean distance between estimated and actual positions

### 2. Signal Processing Algorithms
- **Savitzky-Golay filter**: Smoothing while preserving peak shape
- **Normalization**: Scaling intensity values to [0,1] range
- **Weighted centroid**: Robust distance estimation from noisy waveforms

### 3. Software Architecture
- **Modular design**: Separation of concerns for maintainability
- **Type hints**: Python type annotations for code clarity
- **Error handling**: Graceful fallbacks for edge cases

### 4. Visualization Tools
- **Waveform plots**: Display sensor intensity vs distance
- **Configuration plots**: Show sensor positions and estimated object location
- **Error analysis**: Distribution and trends in estimation errors

---

## Why This Approach Was Chosen

### 1. Distance Extraction Method: Weighted Centroid
**Rationale:**
- More robust to noise than simple peak detection
- Faster than Gaussian fitting
- Accounts for waveform shape, not just maximum
- Formula: `distance = Σ(x_i × y_i) / Σ(y_i)`

### 2. Preprocessing: Savitzky-Golay Filter
**Rationale:**
- Preserves peak location better than moving average
- Reduces high-frequency noise while maintaining signal features
- Standard technique in signal processing

### 3. Trilateration: Least-Squares Optimization
**Rationale:**
- Handles inconsistent measurements from noise
- More accurate than simple algebraic solution for real-world data
- Uses closed-form solution as initial guess for faster convergence

### 4. Modular Architecture
**Rationale:**
- Each component independently testable
- Easy to swap algorithms (e.g., different distance estimation methods)
- Clear separation between I/O, processing, and computation
- Professional software engineering practice

---

## Results Summary

**Implementation Status:**
- ✅ Python program written
- ✅ Uses distance-intensity waveforms (not reference 'd' directly)
- ✅ Modular code structure
- ✅ Multiple distance estimation methods
- ✅ Trilateration for position calculation
- ✅ Visualization capabilities
- ✅ Unit tests
- ✅ Documentation (README, this file)
- ✅ Further exercise considerations discussed

**Current Performance:**
- Mean error: ~190mm (above 100mm target)
- Issue: Error is calculated from center (0,0), not actual object positions

**Note:** The main.py currently compares estimated positions to the triangle center (0,0), not the actual ground truth positions. The ground_truth.json file contains the true object positions which should be used for proper error calculation.

---

## Recommendations for Improvement

1. **Fix error calculation**: Compare to actual positions from ground_truth.json
2. **Parameter tuning**: Adjust smoothing window size and threshold values
3. **Method comparison**: Test peak vs weighted_centroid vs gaussian on actual data
4. **Confidence weighting**: Weight trilateration by signal quality scores

---

## Academic Integrity Statement

This solution was developed with AI assistance (Claude Code by Anthropic) as permitted by the assignment guidelines. The AI was used for:
- Code structure recommendations
- Algorithm implementation assistance
- Documentation generation
- Requirements verification

All prompts used have been documented in this file as required.

---

**Date:** November 2025
**Project:** AI Sensor Exercise - 3-Radar Position Estimation
**Tools Used:**  Python 3.x, NumPy, SciPy, Matplotlib
