/**
 * The flood-depth emulator in the browser.
 *
 * Mirrors manyfews_core/emulator.py. Each of the 302,748 cells stores a cubic in
 * river flow plus its own threshold, so the whole inundation surface follows
 * from one number:
 *
 *     depth(Q) = max(0, P0 + Q*(P1 + Q*(P2 + Q*P3)))   if Q >= minQ, else 0
 *
 * The input is clamped at meta.q_cap_m3s. Beyond the range the cubics were
 * fitted over they diverge - 118 m at Q=500, 1122 m at Q=800 - and the wet-cell
 * count actually falls as they turn over and go negative.
 */

/** Fetch and gunzip a binary asset. */
async function fetchGzip(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  if (typeof DecompressionStream === 'undefined') {
    throw new Error(
      'This browser lacks DecompressionStream. Chrome 80+, Firefox 113+ or Safari 16.4+ is needed.'
    );
  }
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Unpack a little-endian bitset into a Uint8Array of 0/1. */
function unpackBits(bytes, n) {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (bytes[i >> 3] >> (i & 7)) & 1;
  }
  return out;
}

export class FloodEmulator {
  constructor(meta, position, beta, minQ, channel, monotone) {
    this.meta = meta;
    this.n = meta.n_cells;
    this.position = position;   // Uint32Array - index into the display grid
    this.beta0 = beta[0];
    this.beta1 = beta[1];
    this.beta2 = beta[2];
    this.beta3 = beta[3];
    this.minQ = minQ;           // Float32Array, decoded from the level table
    this.channel = channel;     // Uint8Array 0/1
    this.monotone = monotone;   // Uint8Array 0/1
    this.qCap = meta.q_cap_m3s;
    this.nonMonotone = [];
    for (let i = 0; i < this.n; i++) if (!monotone[i]) this.nonMonotone.push(i);
  }

  static async load(base = 'data') {
    const meta = await (await fetch(`${base}/meta.json`)).json();
    const n = meta.n_cells;

    const [grid, channelBits, monotoneBits] = await Promise.all([
      fetchGzip(`${base}/grid.bin.gz`),
      fetchGzip(`${base}/channel.bin.gz`),
      fetchGzip(`${base}/monotone.bin.gz`),
    ]);

    // Column-wise layout, so each block maps straight onto a typed array.
    const buf = grid.buffer;
    let offset = grid.byteOffset;
    const position = new Uint32Array(buf.slice(offset, offset + n * 4));
    offset += n * 4;

    const beta = [];
    for (let b = 0; b < 4; b++) {
      beta.push(new Float32Array(buf.slice(offset, offset + n * 4)));
      offset += n * 4;
    }

    const codes = new Uint8Array(buf.slice(offset, offset + n));
    const levels = meta.min_q_levels;
    const minQ = new Float32Array(n);
    for (let i = 0; i < n; i++) minQ[i] = levels[codes[i]];

    return new FloodEmulator(
      meta, position, beta, minQ,
      unpackBits(channelBits, n),
      unpackBits(monotoneBits, n)
    );
  }

  /** Depth in every cell at a single flow. Returns a Float32Array(n). */
  depthAt(flow, out) {
    const depth = out || new Float32Array(this.n);
    const q = Math.min(flow, this.qCap);
    const { beta0, beta1, beta2, beta3, minQ, n } = this;

    for (let i = 0; i < n; i++) {
      if (flow < minQ[i]) {
        depth[i] = 0;
        continue;
      }
      const d = beta0[i] + q * (beta1[i] + q * (beta2[i] + q * beta3[i]));
      depth[i] = d > 0 ? d : 0;
    }
    return depth;
  }

  /**
   * Depth at one percentile of a pooled flow population.
   *
   * Uses the same exact shortcut as the Python implementation: for a cell whose
   * clamped response is monotone in Q, sorting depths is sorting flows, so the
   * percentile of the depths is the depth at the percentile of the flows -
   * interpolated between the two bracketing order statistics, matching NumPy's
   * default. The ~3% of cells that are not monotone are evaluated in full.
   *
   * @param {Float64Array} sortedFlows ascending pooled samples
   * @param {number} pct percentile in 0..100
   */
  depthPercentile(sortedFlows, pct, out) {
    const M = sortedFlows.length;
    const pos = ((M - 1) * pct) / 100;
    const k = Math.floor(pos);
    const frac = pos - k;
    const lo = sortedFlows[k];
    const hi = sortedFlows[Math.min(k + 1, M - 1)];

    const depthLo = this.depthAt(lo);
    const depthHi = this.depthAt(hi, out || new Float32Array(this.n));
    const depth = depthHi;
    for (let i = 0; i < this.n; i++) {
      depth[i] = depthLo[i] + frac * (depth[i] - depthLo[i]);
    }

    // Exact correction for the cells where the shortcut does not hold.
    const { beta0, beta1, beta2, beta3, minQ, qCap } = this;
    const scratch = new Float64Array(M);
    for (const i of this.nonMonotone) {
      const b0 = beta0[i], b1 = beta1[i], b2 = beta2[i], b3 = beta3[i], mq = minQ[i];
      for (let j = 0; j < M; j++) {
        const f = sortedFlows[j];
        if (f < mq) { scratch[j] = 0; continue; }
        const q = f < qCap ? f : qCap;
        const d = b0 + q * (b1 + q * (b2 + q * b3));
        scratch[j] = d > 0 ? d : 0;
      }
      scratch.sort();
      depth[i] = scratch[k] + frac * (scratch[Math.min(k + 1, M - 1)] - scratch[k]);
    }

    return depth;
  }

  /** Count cells wetter than a threshold, ignoring the river channel. */
  wetCells(depth, threshold = 0.01) {
    let count = 0;
    for (let i = 0; i < this.n; i++) {
      if (!this.channel[i] && depth[i] > threshold) count++;
    }
    return count;
  }

  maxDepth(depth) {
    let max = 0;
    for (let i = 0; i < this.n; i++) {
      if (!this.channel[i] && depth[i] > max) max = depth[i];
    }
    return max;
  }

  meanWetDepth(depth, threshold = 0.01) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < this.n; i++) {
      if (!this.channel[i] && depth[i] > threshold) { sum += depth[i]; count++; }
    }
    return count ? sum / count : 0;
  }
}
