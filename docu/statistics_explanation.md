# Distance Analysis (M3C2 Output)

## Fundamental Metrics

* **Total Count**: Total number of distance values (including NaN). Provides dataset size context.

  ```python
  total_count = len(distances)
  ```

* **NaN Count**: Number of invalid results (e.g., no neighbors). Lower values indicate better coverage.

  ```python
  nan_count = int(np.isnan(distances).sum())
  ```

  * `np.isnan()` returns a Boolean array (`True` where NaN).
  * `.sum()` counts the number of NaN values.

* **% NaN**: Proportion of invalid results. High values indicate poor coverage.

  ```python
  perc_nan = (nan_count / total_count) if total_count > 0 else np.nan
  ```

* **% Valid**: Proportion of valid results (= 1 − %NaN). High values indicate robust coverage.

  ```python
  perc_valid = (1 - nan_count / total_count) if total_count > 0 else np.nan
  ```

* **Valid Count**: Number of valid (non-NaN) values. Comparable to %Valid.

  ```python
  valid = distances[~np.isnan(distances)]
  clipped = valid[(valid >= data_min) & (valid <= data_max)]
  valid_count = int(clipped.size)
  ```

  * `range_override`: Optional tuple to explicitly set the value range.
  * If not set, `data_min`/`data_max` are computed from the data.

* **Valid Sum**: Sum of all valid distances.

  * \~0 → deviations cancel out → no systematic bias between clouds.
  * Positive → comparison surface is higher/outward vs. reference.
  * Negative → comparison surface is lower/inward vs. reference.

  ```python
  valid_sum = float(np.sum(clipped))
  ```

* **Valid Squared Sum**: Sum of squared valid distance values.

  * Each distance $d_i$ is squared ($d_i^2$), then summed.
  * Always ≥ 0.
  * Sensitive to large outliers.

  ```python
  valid_squared_sum = float(np.sum(clipped ** 2))
  ```

---

## Parameters

* **Normal Scale**: Defines the radius (in point cloud units) used for local normal estimation.

  * Too small → surface noise dominates, normals become unstable.
  * Too large → over-smoothing, local detail lost.

  ```python
  normal_scale
  ```

* **Search Scale**: Defines the radius of the projection cylinder along the normal.

  * Typically \~2 × Normal Scale (rule of thumb).
  * Too small → few/no hits → many NaNs.
  * Too large → strong smoothing, loss of detail.

  ```python
  search_scale
  ```

---

## Location & Dispersion Metrics

* **Min / Max**: Minimum and maximum distance values. May reveal extreme outliers.

  ```python
  min_val = float(np.nanmin(distances))
  max_val = float(np.nanmax(distances))
  ```

* **Mean (Bias)**: Arithmetic mean. Ideal near 0.

  ```python
  avg = float(np.mean(clipped))
  ```

* **Median**: Robust central tendency, less sensitive to outliers.

  ```python
  med = float(np.median(clipped))
  ```

* **Empirical Std**: Standard deviation. Sensitive to outliers.

  ```python
  std_empirical = float(np.std(clipped))
  ```

* **RMS (Root Mean Square)**:

  * Square root of the mean squared distances.
  * Reflects typical deviation, includes both spread (Std) and bias (Mean).


    $x_{RMS} = \sqrt{\frac{1}{n} \left( x_{1}^{2} + x_{2}^{2} + \cdots + x_{n}^{2} \right)}$

  ```python
  rms = float(np.sqrt(np.mean(clipped ** 2)))
  ```

* **MAE (Mean Absolute Error)**:

  * $\mathrm{MAE} = \frac{\sum_{i=1}^{n} \lvert y_i - x_i \rvert}{n} 
= \frac{\sum_{i=1}^{n} \lvert e_i \rvert}{n}$
  * Uses absolute values instead of squares.
  * More robust to outliers.
  * Represents the average magnitude of deviations (typical offset between clouds).
  * $MAE = 0$ → perfect agreement.
  * $MAE = 0.01$ → on average, the point clouds deviate by 1 cm (assuming units are in meters).

  ```python
  mae = float(np.mean(np.abs(clipped)))
  ```

* **NMAD (Normalized Median Absolute Deviation)**:

  * Robust estimate of σ.
  * Equivalent to Std for normally distributed data.

  ```python
  mad = float(np.median(np.abs(clipped - med)))
  nmad = float(1.4826 * mad)
  ```

---

## Inlier/Outlier Analysis

* **Inlier/Outlier Definition**: Values classified using threshold $|x| ≤ 3·RMS$.

* **MAE Inlier**: Mean absolute error excluding outliers.

  ```python
  mae_in = float(np.mean(np.abs(inliers))) if inliers.size > 0 else np.nan
  ```

* **NMAD Inlier**: NMAD excluding outliers.

  ```python
  nmad_in = (
      float(1.4826 * np.median(np.abs(inliers - median)))
      if inliers.size > 0 else np.nan
  )
  ```

* **Outlier Count**: Number of points with |x| > 3·RMS.

* **Inlier Count**: Number of points with |x| ≤ 3·RMS.

  * Sum of inliers + outliers = valid\_count.

* **Mean/Std Inlier**: Mean and standard deviation without outliers.

* **Mean/Std Outlier**: Mean and standard deviation of outliers only.

* **Positive/Negative Outliers**: Count of outliers above/below zero.

* **Positive/Negative Inliers**: Counts for inliers above/below zero.

---

## Quantiles

* **Q05/Q95**: Range containing 90% of values.
* **Q25/Q75**: Interquartile range (IQR = Q75 − Q25). Robust spread measure.



## Distribution Fits


### Gaussian Mean / Standard Deviation

* Useful for comparing the distance distribution with a theoretical Gaussian distribution.
* **Gaussian Mean (`mu`)**

  * Location parameter of the fitted normal distribution.
  * Represents the estimated “center” of the data.
  * May slightly differ from the empirical mean since it is obtained through curve fitting.

  ```python
  float(mu)
  ```
* **Gaussian Std (`std`)**

  * Scale parameter of the fitted normal distribution.
  * Represents the estimated standard deviation of the data.
  * Smooths the actual distribution and is less sensitive to small irregularities.

  ```python
  float(std)
  ```
* Example usage:

  ```python
  mu, std = norm.fit(clipped)
  ```
* Library:

  ```python
  from scipy.stats import norm
  ```

---

### Gaussian Chi²

* **Low value** → Histogram matches the Gaussian distribution well → “good fit.”
* **High value** → Strong deviations (e.g., skewness, multimodality, heavy tails).
* The absolute magnitude depends on the number of bins and data size, so it is more suitable for **comparing different runs** rather than absolute interpretation.

**Computation steps:**

1. **Prepare histogram fit:**

   ```python
   cdfL = norm.cdf(bin_edges[:-1], mu, std)
   cdfR = norm.cdf(bin_edges[1:], mu, std)
   ```

2. **Expected frequencies under the Gaussian fit:**

   * Number of points predicted by the fitted distribution in each histogram bin.

   ```python
   expected_gauss = N * (cdfR - cdfL)
   ```

3. **Filter out very small expected values (to avoid division by zero):**

   ```python
   eps = 1e-12
   thr = min_expected if min_expected is not None else eps
   maskG = expected_gauss > thr
   ```

   * Bins with very small expected counts (< thr) are ignored, since they would make the result unstable.

4. **Pearson Chi² statistic:**

   ```python
   pearson_gauss = float(
       np.sum((hist[maskG] - expected_gauss[maskG]) ** 2 / expected_gauss[maskG])
   )
   ```

   * `hist` = observed frequencies (actual counts per bin).
   * `expected_gauss` = expected frequencies under the fitted Gaussian model.

---

## Weibull Distribution

* **Reference**: [Wikipedia](https://en.wikipedia.org/wiki/Weibull_distribution)
* **Library**: [`scipy.stats.weibull_min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html)

The Weibull distribution is often a better fit than a Gaussian distribution for strongly skewed error distributions (e.g., when distances are not normally distributed).

It is one of the most widely used distributions for modeling non-normally distributed data. The Weibull distribution is defined by three parameters: **shape**, **scale**, and **shift (location)**. Depending on their values, the Weibull distribution can take on very different forms.

(Source: [Minitab Documentation](https://support.minitab.com/de-de/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/supporting-topics/distributions-and-transformations-for-nonnormal-data/why-is-weibull-the-default-distribution/))

---

### Weibull Shape Parameter (a)

* **Effect of shape (a):**

  * $a < 1$ → heavy tails, strong right skew.
  * $a \approx 2$ → similar to a Rayleigh distribution.
  * $a > 3$ → distribution becomes more symmetric, approaching Gaussian.
  * ![alt text](image.png)

```python
float(a)
a, loc, b = weibull_min.fit(clipped)
```

---

### Weibull Scale Parameter (b)

* **Effect of scale (b):**

  * Larger values → broader distribution.
  * Roughly corresponds to a “stretching” of the distance distribution.
  * ![alt text](image-1.png)

```python
float(b)
a, loc, b = weibull_min.fit(clipped)
```

---

### Weibull Shift Parameter (loc)

* **Effect of shift (loc):**

  * Translates the distribution along the axis.
  * In CloudCompare, often close to the median or minimum, depending on the dataset.
  * ![alt text](image-2.png)

```python
float(loc)
a, loc, b = weibull_min.fit(clipped)
```

---

### Weibull Mode

* Position of the maximum of the probability density function.

```python
mode_weibull = float(loc + b * ((a - 1) / a) ** (1 / a)) if a > 1 else float(loc)
```

---

### Weibull Skewness

* Measure of asymmetry:

  * Positive = right skew (long right tail).
  * Negative = left skew (long left tail).

```python
skew_weibull = float(weibull_min(a, loc=loc, scale=b).stats(moments="s"))
```

---

### Weibull Chi² (Goodness-of-Fit)

* Pearson Chi² statistic for the Weibull fit.
* Should only be interpreted **relatively** (for comparisons between runs).

**Computation:**

1. Expected frequencies under the Weibull distribution:

   ```python
   cdfL = weibull_min.cdf(bin_edges[:-1], a, loc=loc, scale=b)
   cdfR = weibull_min.cdf(bin_edges[1:], a, loc=loc, scale=b)
   expected_weib = N * (cdfR - cdfL)
   ```

2. Exclude bins with very small expected counts:

   ```python
   maskW = expected_weib > thr
   ```

3. Pearson Chi² statistic:

   ```python
   pearson_weib = float(
       np.sum((hist[maskW] - expected_weib[maskW]) ** 2 / expected_weib[maskW])
   )
   ```

## Distribution Characteristics

* **Skewness**: Asymmetry of distribution.

  * \~0 → symmetric.
  * > 0 → right-skew.
  * <0 → left-skew.

* **Kurtosis (Excess)**: Tail heaviness/peakedness.

  * 0 = Gaussian.
  * > 0 = heavy tails.
  * <0 = light tails.

---

## Tolerance & Coverage Metrics

* **% |Distance| > Threshold**: Fraction exceeding tolerance (e.g., 1 cm).
* **% Within ±2 Std**: Fraction within two standard deviations. \~95% for Gaussian data.
* **Max |Distance|**: Largest absolute deviation (sensitive to outliers).
* **Within Tolerance**: Fraction of values within defined tolerance (default ±1 cm).

---

## Agreement & Comparison Metrics


### Bland–Altman Analysis (Lower/Upper Limits)

* **References**:

  * [Wikipedia: Bland–Altman plot](https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot)
  * [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4470095/)
  * [Datatab Tutorial](https://datatab.net/tutorial/bland-altman-plot)

A **Bland–Altman plot** is used to compare two measurement methods or outputs.
It answers the question:
**“Do both methods provide comparable results?”**

---

#### Structure of the Diagram
![alt text](image-3.png)
* **x-axis (Mean of measurements)**

  * Mean of both methods per data point.
  * Shows how large the measured values are on average.

* **y-axis (Difference)**

  * Difference between the two methods per data point (typically Method A − Method B).
  * Shows how large the deviations are and in which direction.

* **Red line (Bias)**

  * Mean difference across all points.
  * Indicates the **systematic offset** between the methods.

* **Green lines (Limits of Agreement)**

  * Bias ± 1.96 × standard deviation of the differences.
  * Range in which \~95% of all differences are expected to lie.
  * Provides a measure of the **spread of deviations**.

---

#### Interpretation

1. **Bias close to 0** → Methods are, on average, equivalent.
   **Bias significantly ≠ 0** → One method consistently yields higher/lower results.

2. **Narrow Limits of Agreement** → High agreement (low variance).
   **Wide Limits** → High uncertainty, poor reproducibility.

3. **Pattern of the scatter plot:**

   * Random scatter around the bias line → Differences are independent of measurement magnitude (good).
   * Systematic structures or trends (e.g., funnel shape, slope) → Deviations depend on the size of the measurement (heteroscedasticity, systematic errors).

---

#### What a Bland–Altman Plot Shows

* The **systematic bias** between two outputs.
* The **spread of deviations**.
* Whether the differences are **random** or depend on the magnitude of the measured values.


### Passing-Bablok Regression
Passing–Bablok regression is a non-parametric way to, for example, compare two methods of measuring something in order to judge if they are equivalent or not. It was first described in Passing & Bablok (1983). 

[Reference H. Passing und W. Bablok](https://www.degruyterbrill.com/document/doi/10.1515/cclm.1983.21.11.709/html)

[Code Example](https://rowannicholls.github.io/python/statistics/hypothesis_testing/passing_bablok.html)

[Wikipedia](https://rowannicholls.github.io/python/statistics/hypothesis_testing/passing_bablok.html)

Essentially, when two different methods are being used to take the same measurements the values generated can be plotted on a scatter plot and a line-of-best-fit can fitted. The Passing-Bablok method does this fitting by:

+ Looking at every combination of two points in the scatter plot
+ Drawing a line between the two points in each of these combinations
+ Taking the gradient of each of these lines
+ Extending these lines to the y-axis and taking the y-intercepts
+ Taking the median of the gradients and the median of the y-intercepts of these lines

The median gradient and median y-intercept create the overall line-of-best-fit. This has an associated confidence interval which can be interpreted as follows:

+ If 1 is within the confidence interval of the gradient and 0 is within the confidence interval of the y-intercept, then the two methods are comparable within the investigated concentration range
+ If 1 is not within the confidence interval of the gradient then there is a proportional difference between the two methods
+ If 0 is not within the confidence interval of the y-intercept then there is a systematic difference.
+ ![alt text](image-4.png)


# Single-Cloud Statistics

### Input / Context Parameters


* **Radius \[m]** – Search radius for local neighborhoods (sphere in 3D).

  * Larger radius = smoother, less noise-sensitive metrics.
  * Smaller radius = more detailed, but also more sensitive to noise.

* **k-NN** – Number of nearest neighbors used for k-NN distances.

  * Larger k = more stable, but less localized.

* **Sampled Points** – Number of points randomly sampled for local metrics (kNN, radius, PCA).

  * Controls runtime and stability.

* **Area Source** – Method used to estimate XY area:

  * `convex_hull`: convex hull of the footprint (closer to reality).
  * `bbox`: axis-aligned bounding box (upper bound).

* **Area XY \[m²]** – Estimated footprint area in XY (used for global density calculations).

---

### Global Height Statistics (Z)


* **Z Min / Max \[m]** – Minimum and maximum elevation values.
  *   $z_{\min}=\min(z),\quad z_{\max}=\max(z)$
  * Interpretation: Coarse elevation extremes; may include outliers.

* **Z Mean / Median \[m]** – Central tendency of elevation values.

  * Median is more robust to outliers.
  * Difference between mean and median indicates skewness.
  * $\bar z=\frac{1}{N}\sum_i z_i,\quad \tilde z=\mathrm{median}(z)$

* **Z Std \[m]** – Standard deviation of elevation values.

  * Large = strong relief variation;
  * Small = flat/homogeneous surface.
  * $\sigma_z=\sqrt{\tfrac{1}{N}\sum_i (z_i-\bar z)^2}$

* **Z Quantiles (Q05, Q25, Q75, Q95) \[m]** – Elevation quantiles.

  * Provides range without extreme outliers.
  * Interquartile Range (IQR = Q75 − Q25) describes relief width.
  * `np.percentile(z, [5, 25, 75, 95])`

---

### Area & Global Point Density

Let $xy = P[:,0:2]$.

* **Area XY \[m²]** – Footprint area (see above).
* **Density Global \[pt/m²]** – Number of points per unit area.

  * High = dense sampling;
  * Low = sparse coverage.
  * Dependent on the chosen `Area Source`.
  * $\rho_{\text{global}}=\frac{N}{A_{\text{XY}}}$

---


### k-Nearest Neighbors (kNN)
On subsample $S$ (size $M\le N$):

* **Mean NN Dist (1..k) \[m]** – Mean distance to neighbors 1 through k.
  * $\overline{d}_{1..k}=\frac{1}{M}\sum_{p\in S}\frac{1}{k}\sum_{j=1}^{k} d_j(p)$
  * Code: `mean_nn_all = np.mean(dists_knn[:, 1:])` (index 0 is the point itself)

  * Small = dense sampling;
  * Large = sparse sampling or gaps.

* **Mean Distance to k-th NN \[m]** – Average distance to the k-th neighbor.
  * $\overline{d}_{k}=\frac{1}{M}\sum_{p\in S} d_k(p)$
  * Code: `mean_nn_kth = np.mean(dists_knn[:, min(k, dists_knn.shape[1]-1)])`

  * Acts as a scale indicator.
  * Smaller = denser

---

### Radius-Based Neighborhoods (per point $p\in S$)

Neighbors: $\mathcal{N}_r(p)=\{q\in S:\lVert q-p\rVert\le r\}$, volume $V=\tfrac{4}{3}\pi r^3$.

* **Local Density \[pt/m³] (Mean/Median/Q05/Q95)** – Number of neighbors divided by neighborhood volume.

  * High = dense sampling;
  * Low = sparse sampling.
  * Quantiles describe heterogeneity of density.
  * $\rho(p)=\frac{|\mathcal{N}_r(p)|}{V}$

* **Roughness \[m] (Mean/Median/Q05/Q95)** – Standard deviation of perpendicular distances of neighbors to the local PCA plane.

  * Low = smooth/planar surface.
  * High = rough, uneven, or noisy surface. 

---

### PCA Shape Descriptors

Let eigenvalues sorted $\lambda_1\ge \lambda_2\ge \lambda_3\ge 0$ of covariance of $U$.

Derived from eigenvalues of local PCA analysis of point neighborhoods:

* **Linearity** – High when structure is line-like (e.g., an edge).
  * $\mathrm{Lin}=\frac{\lambda_1-\lambda_2}{\lambda_1}$
  * Code: `(evals[0] - evals[1]) / evals[0]`
  
* **Planarity** – High when structure is planar (e.g., flat surface).
  * $\mathrm{Pla}=\frac{\lambda_2-\lambda_3}{\lambda_1}$
  * Code: `(evals[1] - evals[2]) / evals[0]`

* **Sphericity** – High when distribution is isotropic; low when strongly structured.
  * $\mathrm{Sph}=\frac{\lambda_3}{\lambda_1}$
  * Code: `evals[2] / evals[0]`
  
* **Anisotropy** – High when variance is strongly directional (line or plane).
  * $\mathrm{Ani}=\frac{\lambda_1-\lambda_3}{\lambda_1}$
  * Code: `(evals[0] - evals[2]) / evals[0]`
   
* **Omnivariance \[m²]** – Overall spread of variance (scale-dependent).
  * $\mathrm{Omni}=(\lambda_1\lambda_2\lambda_3)^{1/3}$
  * Code: `np.cbrt(np.prod(evals))`

* **Eigenentropy** – Higher when distribution is disordered/isotropic; lower when structured.
  * With $s=\lambda_1+\lambda_2+\lambda_3$, $p_i=\lambda_i/s$:
  * $H=-\sum_{i=1}^{3} p_i \log(p_i)$
  * Code: `-np.sum(ratios * np.log(ratios + 1e-15))`
  
* **Curvature** – High = strong curvature/relief; low = flat.
  * \kappa=\frac{\lambda_3}{\lambda_1+\lambda_2+\lambda_3}
  * Code: `curvature = evals[2] / sum_eval`

---

### Orientation Metrics

* **Verticality \[deg] (Mean/Median/Q05/Q95)** – Angle between local normal vector and Z-axis.
  * With local normal $n$ (unit) and global Z‑axis $e_z=(0,0,1)$:
  * $\theta(p)=\arccos\big(|n\cdot e_z|\big)\cdot\frac{180}{\pi}$
  * Code: `np.degrees(np.arccos(np.clip(np.abs(n[2]), -1.0, 1.0)))`

  * 0° → Normal points upward/downward (horizontal surface).
  * 90° → Normal lies horizontal (vertical surface).
  * High = predominantly vertical surfaces.
  * Low = predominantly horizontal surfaces.

* **Normal Std Angle \[deg]** – Standard deviation of the angle between locally fitted normals and their mean direction (sign consistency considered).
  * $\sigma_{\angle}=\mathrm{std}\left(\arccos\big(\mathrm{clip}(N\,\bar n,-1,1)\big)\right)\cdot\frac{180}{\pi}$
  * Code: compute `ang = degrees(arccos(cosang))`, then `np.std(ang)`

  * Low = consistent normals (smooth, uniform orientation).
  * High = inconsistent normals (edges, noise, mixed geometries).
