// Landing background: a Three.js "compute grid" — a waving lattice of glowing
// points (one per imaginary kernel thread) drifting through space, with an
// animated scanline shimmer. Pure points + additive blending, no postprocessing,
// so it stays light and dependency-free beyond three.

import * as THREE from "three";

export function initLanding(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x020203, 0.06);

  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 1.6, 9);

  // --- the grid of points ---
  const GRID = 96;
  const SPAN = 28;
  const count = GRID * GRID;
  const positions = new Float32Array(count * 3);
  const seeds = new Float32Array(count);
  let p = 0;
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      positions[p * 3] = (i / (GRID - 1) - 0.5) * SPAN;
      positions[p * 3 + 1] = 0;
      positions[p * 3 + 2] = (j / (GRID - 1) - 0.5) * SPAN;
      seeds[p] = Math.random() * Math.PI * 2;
      p++;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("seed", new THREE.BufferAttribute(seeds, 1));

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 } },
    vertexShader: /* glsl */ `
      attribute float seed;
      uniform float uTime;
      varying float vGlow;
      void main() {
        vec3 pos = position;
        float r = length(pos.xz);
        // travelling waves + per-point shimmer
        float wave = sin(r * 0.6 - uTime * 1.4) * 0.6
                   + sin(pos.x * 0.4 + uTime * 0.8) * 0.3
                   + sin(seed + uTime * 2.0) * 0.15;
        pos.y = wave;
        vGlow = smoothstep(-0.5, 1.0, wave);
        vec4 mv = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = (2.0 + vGlow * 3.5) * (12.0 / -mv.z);
      }
    `,
    fragmentShader: /* glsl */ `
      varying float vGlow;
      void main() {
        vec2 uv = gl_PointCoord - 0.5;
        float d = length(uv);
        if (d > 0.5) discard;
        float a = smoothstep(0.5, 0.0, d);
        // purple -> cyan by glow, matching the page accents
        vec3 cold = vec3(0.45, 0.32, 0.95);
        vec3 hot  = vec3(0.39, 1.0, 0.78);
        vec3 col = mix(cold, hot, vGlow);
        gl_FragColor = vec4(col, a * (0.25 + vGlow * 0.75));
      }
    `,
  });

  const points = new THREE.Points(geo, mat);
  scene.add(points);

  // a faint horizon glow plane
  const glowGeo = new THREE.PlaneGeometry(80, 30);
  const glowMat = new THREE.MeshBasicMaterial({ color: 0x140a2e, transparent: true, opacity: 0.5 });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.position.set(0, 3, -14);
  scene.add(glow);

  function resize() {
    const w = canvas.clientWidth || canvas.parentElement.clientWidth;
    const h = canvas.clientHeight || canvas.parentElement.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas.parentElement);
  resize();

  let raf = 0;
  let mx = 0, my = 0;
  const onMove = (e) => {
    mx = (e.clientX / innerWidth - 0.5);
    my = (e.clientY / innerHeight - 0.5);
  };
  addEventListener("pointermove", onMove);

  const clock = new THREE.Clock();
  function frame() {
    const t = clock.getElapsedTime();
    mat.uniforms.uTime.value = t;
    // gentle parallax + slow orbit
    camera.position.x += (mx * 3 - camera.position.x) * 0.03;
    camera.position.y += (1.6 - my * 2 - camera.position.y) * 0.03;
    camera.lookAt(0, 0.3, 0);
    points.rotation.y = Math.sin(t * 0.05) * 0.15;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  frame();

  return {
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      removeEventListener("pointermove", onMove);
      geo.dispose();
      mat.dispose();
      glowGeo.dispose();
      glowMat.dispose();
      renderer.dispose();
    },
  };
}
