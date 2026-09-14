/**
 * Atoms: pure path builders. No colour, no state, no knowledge of the spec.
 *
 * The layering is the same one the Python registry declares — atoms, then
 * semantic primitives, then macros — and it exists so a primitive can be restyled
 * or animated without touching geometry, and a shape can be fixed without
 * touching every primitive that uses it.
 *
 * ANGLE CONVENTION: degrees, clockwise from +x, in stage coordinates where +y
 * points down. That is canvas's own convention, so an angle converts to radians
 * and draws with no flip. A physics document that means "downwards" by -90 is
 * using the opposite (y-up) convention; reconciling the two is the layout pass's
 * job, not this file's.
 */

/** Degrees to radians. Named so the convention above is visible at every call site. */
export function rad(degrees) {
  return (degrees * Math.PI) / 180;
}

export function circlePath(ctx, cx, cy, radius) {
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.closePath();
}

/**
 * A rounded rectangle, traced by hand rather than via `ctx.roundRect`.
 *
 * `roundRect` is recent enough (Chrome 99 / Safari 16.4) that a canvas without it
 * fails by drawing a *square* — a silent shape change, not an error.
 */
export function roundRectPath(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  const left = x - width / 2;
  const top = y - height / 2;
  const right = left + width;
  const bottom = top + height;
  ctx.beginPath();
  ctx.moveTo(left + r, top);
  ctx.lineTo(right - r, top);
  ctx.quadraticCurveTo(right, top, right, top + r);
  ctx.lineTo(right, bottom - r);
  ctx.quadraticCurveTo(right, bottom, right - r, bottom);
  ctx.lineTo(left + r, bottom);
  ctx.quadraticCurveTo(left, bottom, left, bottom - r);
  ctx.lineTo(left, top + r);
  ctx.quadraticCurveTo(left, top, left + r, top);
  ctx.closePath();
}

/**
 * A regular polygon, or a star when `innerRatio` is below 1.
 *
 * The star case is not decoration: the quality baseline draws every obstacle as
 * 8 vertices alternating between `r` and `0.78r` (`frontend/demo/app.js:696`).
 * That is two numbers, so it stays an atom instead of becoming a glyph asset.
 */
export function polygonPath(ctx, cx, cy, radius, sides, rotation = 0, innerRatio = 1) {
  const step = (Math.PI * 2) / sides;
  const count = innerRatio < 1 ? sides * 2 : sides;
  ctx.beginPath();
  for (let index = 0; index < count; index += 1) {
    const scale = innerRatio < 1 && index % 2 === 1 ? innerRatio : 1;
    const angle = rotation + index * (innerRatio < 1 ? step / 2 : step);
    const px = cx + Math.cos(angle) * radius * scale;
    const py = cy + Math.sin(angle) * radius * scale;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

/** A filled pie slice spanning `fov` degrees, centred on `heading`. */
export function sectorPath(ctx, cx, cy, radius, fov, heading) {
  const start = rad(heading - fov / 2);
  const end = rad(heading + fov / 2);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, radius, start, end);
  ctx.closePath();
}

export function polylinePath(ctx, points) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
}

/** A line with a triangular head at (x2, y2). */
export function arrowPath(ctx, x1, y1, x2, y2, head = 10) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const length = Math.hypot(x2 - x1, y2 - y1);
  // Never let the head eat the shaft: a short arrow with a fixed head becomes an
  // arrowhead with no tail, which reads as a different symbol.
  const h = Math.min(head, length * 0.5);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - h * Math.cos(angle - Math.PI / 6), y2 - h * Math.sin(angle - Math.PI / 6));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - h * Math.cos(angle + Math.PI / 6), y2 - h * Math.sin(angle + Math.PI / 6));
}
