# Terminal Weld Inspection Memory

- ROI center must be geometry-first. Use the terminal green box center as the prior, then only search near that center.
- Do not let the brightest column dominate center location. Missing and less samples can be pulled toward solder-ball reflections.
- Current ROI center search uses terminal center +/- 16% terminal width and scores only the upper 60% of the search region.
- Seam center scoring uses multi-scale core/guard/side windows. Dark slots and bright seams are both valid contrast, but the polarity must stay available for later classification.
- The part is not a pure rectangle: upper area is round/large-radius, lower area is the rectangular weld zone. Use the full contour only for coarse angle, then locate the lower rectangular work area from bottom edge and side boundaries.
- ROI bottom should align to the lower work-area bottom edge. Do not extend the red ROI below the lower work area.
- Lower-half angle estimation is more stable than full-part angle, but direct `minAreaRect` on the lower half can collapse to a horizontal box. Use lower-half side-boundary slope refinement instead.
- For missing samples, center search must treat bright slot edges as evidence around a dark valley, not as the seam center itself. Dark-valley score should dominate bright-center score.
- The debug green box is the coarse-aligned lower 50% region used for angle refinement. Keep it visible in overlay for human inspection.
- Direct `minAreaRect` on the lower-half crop is not reliable because the cropped shape can become wider than tall. Current angle refinement uses near-vertical Hough line residuals inside the lower-half crop.
