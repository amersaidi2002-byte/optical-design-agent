# Physics, conventions and provenance

## Coordinates and surfaces

The optical axis is +z, with rays travelling from an object at infinity toward a planar sensor. A spherical surface with vertex z_v and signed radius R has sphere centre z_v + R. The physical cap is

    z(h) = z_v + R − sign(R) sqrt(R² − h²).

The sphere root is chosen by curvature sign; simply taking the smallest positive ray–sphere intersection is incorrect for negative-radius caps. A hit is rejected if its ray parameter is nonpositive, if it misses the common clear aperture, or if refraction is total internal reflection. Refraction must preserve a positive z direction. The sensor must be ahead of the final ray position.

A stop at z=0 is sampled by tracing incident rays from z=−2× pupil radius with the appropriate field-dependent lateral shift. All lenses share a learned clear semi-diameter greater than the stop radius. No additional field-dependent pupil aiming is performed.

## Parameters

Continuous genes u in [0,1] are mapped linearly into these normalized ranges:

| Parameter | Lower | Upper |
|---|---:|---:|
| R1 / target focal length | +0.55 | +3.0 |
| R2 / target focal length | −3.0 | −0.55 |
| R3 / target focal length | −5.0 | −0.8 |
| R4 / target focal length | +0.8 | +5.0 |
| Each centre thickness / target focal length | 0.025 | 0.12 |
| Vertex air gap / target focal length | 0.025 | 0.22 |
| Last-vertex-to-sensor / target focal length | 0.45 | 1.5 |
| Clear semi-diameter / stop radius | 1.05 | 1.8 |

For negative radii, increasing the gene increases |R|; the table lists signed physical ranges. Requirements are sampled over f=25–80 mm, half-FOV=2–10°, f/#=3.5–9. At other requirements the CLI can still run, but no generalization claim is made.

The minimum clearance is computed between adjacent caps at centre and edge of the shared clear disk. For this fixed spherical sign pattern, adjacent-cap sag differences are monotone with radial height, so these endpoints bound their axial clearance. Curvature radii smaller than the clear semi-diameter are rejected mechanically.

## Dispersion and paraxial optics

The Sellmeier form is n²(λ) = 1 + Σ B_i λ² / (λ² − C_i), with λ in µm. The catalog constants come from the manufacturer's data sheets below. Tests check d-line indices and normal visible dispersion.

The paraxial ray state is [height, n×angle]. A translation has B=d/n and a refracting surface has C=−(n_out−n_in)/R. With air on both sides, EFL=−1/C and BFL=−A/C for the system matrix. The test suite checks the thick-lens lensmaker limit and full traced paraxial focus independently of the image-plane objective.

Pupil samples use equal-area annular radii and equal angular weights, plus a centre sample of the same weight. This is a finite approximate quadrature, not an analytic pupil integral. A shared centroid across wavelengths retains lateral chromatic spreading; a separate centroid is used for each field, so distortion is not penalized.

## Primary references

- [PBRT, Realistic Cameras, §6.4](https://pbr-book.org/3ed-2018/Camera_Models/Realistic_Cameras): spherical lens intersections and correct surface branch selection.
- [PBRT, Specular Reflection and Transmission, §8.2](https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission): vector refraction and total internal reflection.
- [SCHOTT optical glass downloads](https://www.schott.com/en-us/products/optical-glass-p1000267/downloads): manufacturer catalog and glass data sheets, including dispersion coefficients.
- [SCHOTT Optical Glass Collection, March 2018](https://www.schott.com/shop/medias/schott-optical-glass-collection-datasheets-english-march2018.pdf): N-BK7, N-F2 and N-SF5 data sheets; Constants of Dispersion Formula.
- [SCHOTT N-F2 catalog entry](https://us.shop.schott.com/advanced-optics/en/Optical-Glass/N-F2/c/glass-N-F2): refractive indices at the sampled spectral lines.

These references support the physical equations/catalog, not the accuracy of this implementation. Local numerical tests provide internal consistency checks. Independent optical-software validation remains future work.
