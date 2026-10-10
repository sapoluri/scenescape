// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Re-export vendored ProjectedMaterial (marcofugaro/three-projected-material)
 * with a Vite-compatible `three` import. Same module as
 * `/static/assets/ProjectedMaterial.module.js` used by the legacy calibrate
 * viewport — keeps camera→floor coverage projection compatible.
 */
export { default as ProjectedMaterial } from "./ProjectedMaterial.module.js";
