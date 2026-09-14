/**
 * Semantic primitives: the 13 T1 kinds from `rendering/registry.py`, drawn.
 *
 * Each drawer takes the same four things — a context, the element, the live view
 * (simulation state) and the theme — and produces only the *shape of the thing*.
 * Colour comes from `tone` through the theme, never from a literal here, because
 * `body.danger` flips during playback and a frozen colour cannot follow it
 * (decision D1).
 *
 * Only the kinds the baseline actually uses are implemented. The rest are
 * declared in `registry.js` and refuse to draw, which is intentional: a
 * half-drawn primitive that silently renders nothing is worse than a loud
 * failure, because the picture still "works".
 */

import {
  arrowPath,
  circlePath,
  polylinePath,
  polygonPath,
  rad,
  roundRectPath,
  sectorPath,
} from "./atoms.js";

/** Font for a primitive's own annotations — tick values, measurements, names. */
const ANNOTATION_FONT = "13px Inter, 'Microsoft YaHei', sans-serif";

/**
 * Draw text centred on a point, in the theme's text colour.
 *
 * `fillText`, never `innerHTML`: a label here is model-authored data (decision
 * D4). The annotation primitives draw their own text because a `dimension` with
 * no number on it is not a drawing of a dimension — unlike a `body`'s name,
 * which is a caption about the object rather than part of its geometry.
 */
function annotation(ctx, view, text, x, y) {
  if (!text) return;
  ctx.save();
  ctx.font = ANNOTATION_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // A dark plate behind the glyphs, or a tick value lands on the shaft it labels.
  const width = ctx.measureText(text).width;
  ctx.fillStyle = view.theme.background;
  ctx.globalAlpha = 0.85;
  ctx.fillRect(x - width / 2 - 3, y - 8, width + 6, 16);
  ctx.globalAlpha = 1;
  ctx.fillStyle = view.theme.text;
  ctx.fillText(text, x, y);
  ctx.restore();
}

/**
 * Where an anchored element starts now, and how far it has drifted.
 *
 * The layout bakes coordinates; the simulation moves the body they were baked
 * from. `attachment` puts the element's own x/y on the body each frame, so the
 * shift is the difference between the live node and the authored one — and it
 * is what a baked polyline has to be translated by, since moving a trace's
 * centre does not move its points.
 */
function drift(element, view) {
  const node = view.live.get(element.id) ?? element;
  return { x: node.x, y: node.y, dx: node.x - element.x, dy: node.y - element.y };
}

/** `tone` -> theme slot. The one place a semantic name becomes a colour. */
function toneColor(theme, tone) {
  switch (tone) {
    case "danger":
      return theme.danger;
    case "accent":
      return theme.yellow;
    case "success":
      return theme.green;
    case "muted":
      return theme.muted;
    default:
      return theme.lineHot;
  }
}

/** Highlighted elements glow and draw brighter; everything else stays quiet. */
function applyHighlight(ctx, theme, highlighted) {
  if (!highlighted) return;
  ctx.shadowColor = theme.lineHot;
  ctx.shadowBlur = 18;
}

export function drawBody(ctx, element, view) {
  if (element.glyph) {
    // Not a fallback: drawing the plain shape instead would look deliberate.
    throw new Error(
      `字形 \`${element.glyph}\` 还没有数据（assets/glyphs/ 为空）；` +
        `body \`${element.id}\` 无法绘制`,
    );
  }
  const node = view.live.get(element.id) ?? element;
  const color = toneColor(view.theme, view.isDangerous(element.id) ? "danger" : element.tone);
  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));

  if (element.shape === "circle") {
    circlePath(ctx, node.x, node.y, element.width / 2);
  } else if (element.shape === "polygon") {
    polygonPath(
      ctx,
      node.x,
      node.y,
      element.width / 2,
      element.sides ?? 6,
      // A point-up polygon, so an octagon sits on a flat edge the way the
      // baseline's obstacles do rather than on a vertex.
      rad(-90 + 180 / (element.sides ?? 6)),
      element.inner_ratio ?? 1,
    );
  } else {
    roundRectPath(ctx, node.x, node.y, element.width, element.height, 8);
  }

  ctx.fillStyle = view.theme.panel;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();

  if (element.props?.show_heading) {
    drawHeading(ctx, node, element, color);
  }
  ctx.restore();
}

/** The little direction triangle the baseline puts on its car (`app.js:732`). */
function drawHeading(ctx, node, element, color) {
  const angle = rad(node.heading ?? 0);
  const reach = (element.width ?? 40) * 0.62;
  const size = 7;
  const tipX = node.x + Math.cos(angle) * reach;
  const tipY = node.y + Math.sin(angle) * reach;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(
    tipX - size * Math.cos(angle - Math.PI / 5),
    tipY - size * Math.sin(angle - Math.PI / 5),
  );
  ctx.lineTo(
    tipX - size * Math.cos(angle + Math.PI / 5),
    tipY - size * Math.sin(angle + Math.PI / 5),
  );
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

export function drawEmitter(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  const radius = view.lookup(element.id, "radius", element.radius);
  const heading = node.heading ?? element.heading ?? 0;
  const fov = element.fov ?? 180;
  const rays = element.rays ?? 13;
  const color = toneColor(view.theme, element.tone);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));

  sectorPath(ctx, node.x, node.y, radius, fov, heading);
  ctx.fillStyle = "rgba(66, 247, 255, 0.08)";
  ctx.fill();

  // The nearest obstacle inside range, which is the ray the baseline singles out.
  const hit = nearestObstacle(node, view, radius, fov, heading);

  ctx.lineWidth = 1;
  const step = rays > 1 ? fov / (rays - 1) : 0;
  for (let index = 0; index < rays; index += 1) {
    const angle = heading - fov / 2 + step * index;
    const reach = radius;
    // A slow shimmer so the fan reads as live. Phase comes from the ray index,
    // not from a clock, so two frames of the same spec still match.
    const alpha = 0.72 + Math.sin(view.time * 8 + index) * 0.18;
    ctx.strokeStyle = color;
    ctx.globalAlpha = Math.abs(alpha);
    ctx.beginPath();
    ctx.moveTo(node.x, node.y);
    ctx.lineTo(node.x + Math.cos(rad(angle)) * reach, node.y + Math.sin(rad(angle)) * reach);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  if (hit) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = view.theme.danger;
    ctx.beginPath();
    ctx.moveTo(node.x, node.y);
    ctx.lineTo(hit.x, hit.y);
    ctx.stroke();
    circlePath(ctx, hit.pointX, hit.pointY, 5);
    ctx.fillStyle = view.theme.danger;
    ctx.fill();
  }
  ctx.restore();
}

/** Nearest obstacle centre within the fan, plus where the ray meets it. */
function nearestObstacle(node, view, radius, fov, heading) {
  let best = null;
  for (const id of view.obstacleIds) {
    const other = view.live.get(id);
    if (!other) continue;
    const distance = Math.hypot(other.x - node.x, other.y - node.y);
    if (distance > radius) continue;
    const bearing = (Math.atan2(other.y - node.y, other.x - node.x) * 180) / Math.PI;
    const delta = Math.abs(((bearing - heading + 540) % 360) - 180);
    if (delta > fov / 2) continue;
    const angle = rad(bearing);
    const edge = Math.max(0, distance - (view.obstacleRadius.get(id) ?? 0));
    const candidate = {
      x: node.x + Math.cos(angle) * edge,
      y: node.y + Math.sin(angle) * edge,
      pointX: other.x,
      pointY: other.y,
      distance,
    };
    if (best === null || distance < best.distance) best = candidate;
  }
  return best;
}

export function drawZone(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  const radius = view.lookup(element.id, "radius", element.radius);
  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.setLineDash([8, 9]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  circlePath(ctx, node.x, node.y, radius);
  ctx.stroke();
  ctx.restore();
}

export function drawLink(ctx, element, view) {
  const active = view.lookup(element.id, "active", element.active);
  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  if (!active) ctx.setLineDash([8, 12]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = active ? view.theme.lineHot : view.theme.muted;
  polylinePath(ctx, element.points);
  ctx.stroke();
  ctx.restore();
}

export function drawTraveler(ctx, element, view) {
  const path = view.elementById.get(element.path_id);
  if (!path || !path.points || path.points.length < 2) return;
  // Progress is simulation state, not a prop: it is computed from elapsed time
  // rather than authored, so it is read off the live node.
  const progress = view.live.get(element.id)?.progress ?? element.progress ?? 0;
  const point = pointAlong(path.points, progress);
  const size = element.size ?? 10;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.shadowColor = view.theme.yellow;
  ctx.shadowBlur = 18;
  roundRectPath(ctx, point.x, point.y, size * 2.2, size * 1.35, 4);
  ctx.fillStyle = view.theme.yellow;
  ctx.fill();
  ctx.restore();
}

export function drawReadout(ctx, element, view) {
  const text = String(view.lookup(element.id, "text", element.text) ?? "");
  const lines = text.split("\n");
  const padding = 12;
  const lineHeight = 20;
  const longest = lines.reduce((max, line) => Math.max(max, line.length), 0);
  const width = element.width || Math.min(320, Math.max(120, longest * 13 + padding * 2));
  const height = element.height || lines.length * lineHeight + padding * 2;
  const left = element.x - width / 2;
  const top = element.y - height / 2;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  roundRectPath(ctx, element.x, element.y, width, height, 10);
  ctx.fillStyle = view.theme.panel;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.fillStyle = view.theme.text;
  ctx.font = "14px Inter, 'Microsoft YaHei', sans-serif";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  lines.forEach((line, index) => {
    // `fillText`, never `innerHTML` — spec text is data, and the player does not
    // hand it to anything that parses (decision D4).
    ctx.fillText(line, left + padding, top + padding + lineHeight * (index + 0.5));
  });
  ctx.restore();
}

export function drawVector(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  const tipX = node.x + element.dx;
  const tipY = node.y + element.dy;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  arrowPath(ctx, node.x, node.y, tipX, tipY, element.head);
  ctx.stroke();
  ctx.restore();

  // Named beyond the tip: which arrow is v₀ and which is gravity is the whole
  // content of the picture, and two unlabelled arrows say nothing.
  const angle = Math.atan2(element.dy, element.dx);
  annotation(ctx, view, element.label, tipX + Math.cos(angle) * 20, tipY + Math.sin(angle) * 20);
}

export function drawTrace(ctx, element, view) {
  const { dx, dy } = drift(element, view);
  const points = element.points.map((point) => ({ x: point.x + dx, y: point.y + dy }));

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 2;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  if (element.dashed) ctx.setLineDash([7, 7]);
  polylinePath(ctx, points);
  ctx.stroke();
  ctx.restore();
}

export function drawAxis(ctx, element, view) {
  const angle = rad(element.heading);
  const tipX = element.x + Math.cos(angle) * element.length;
  const tipY = element.y + Math.sin(angle) * element.length;
  const normalX = -Math.sin(angle) * 6;
  const normalY = Math.cos(angle) * 6;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  arrowPath(ctx, element.x, element.y, tipX, tipY, 9);
  ctx.stroke();

  // Graduations across the shaft, evenly spaced along it. They mark magnitude
  // rather than a value, because the document's units never reached this layer —
  // the axis says "this far is further than that far", which is what the
  // comparisons the lesson draws actually rest on.
  ctx.beginPath();
  for (let index = 1; index <= element.ticks; index += 1) {
    const along = element.length * (index / element.ticks);
    const px = element.x + Math.cos(angle) * along;
    const py = element.y + Math.sin(angle) * along;
    ctx.moveTo(px - normalX, py - normalY);
    ctx.lineTo(px + normalX, py + normalY);
  }
  ctx.stroke();
  ctx.restore();

  annotation(ctx, view, element.label, tipX + Math.cos(angle) * 18, tipY + Math.sin(angle) * 18);
}

export function drawDimension(ctx, element, view) {
  // Between two objects it re-resolves every frame, so the callout stays on the
  // thing it names when the thing moves. Between two fixed points it does not.
  const start = (element.from_id && view.live.get(element.from_id)) || element.start;
  const end = (element.to_id && view.live.get(element.to_id)) || element.end;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  // Heads at both ends, which is what makes a line read as a measurement
  // rather than as a path something travels along.
  arrowPath(ctx, start.x, start.y, end.x, end.y, 9);
  ctx.stroke();
  arrowPath(ctx, end.x, end.y, start.x, start.y, 9);
  ctx.stroke();
  ctx.restore();

  const midX = (start.x + end.x) / 2;
  const midY = (start.y + end.y) / 2;
  annotation(ctx, view, element.label, midX, midY - 16);
}

export function drawAngle(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  // Physics angles are y-up and the stage is y-down; `layout.py` converts for
  // the vectors it bakes into dx/dy, and this is the same conversion applied to
  // the two directions this element carries as numbers.
  const from = rad(-element.from_degrees);
  const to = rad(-element.to_degrees);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);

  ctx.beginPath();
  ctx.moveTo(node.x, node.y);
  ctx.lineTo(node.x + Math.cos(from) * element.radius, node.y + Math.sin(from) * element.radius);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(node.x, node.y, element.radius, from, to, to < from);
  ctx.stroke();
  ctx.restore();

  const mid = (from + to) / 2;
  annotation(
    ctx,
    view,
    element.label || `${Math.round(element.to_degrees)}°`,
    node.x + Math.cos(mid) * (element.radius + 18),
    node.y + Math.sin(mid) * (element.radius + 18),
  );
}

/** The point `progress` of the way along a polyline, in stage coordinates. */
export function pointAlong(points, progress) {
  const clamped = Math.max(0, Math.min(1, progress));
  const lengths = [];
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    const segment = Math.hypot(
      points[index].x - points[index - 1].x,
      points[index].y - points[index - 1].y,
    );
    lengths.push(segment);
    total += segment;
  }
  let target = total * clamped;
  for (let index = 0; index < lengths.length; index += 1) {
    if (target <= lengths[index] || index === lengths.length - 1) {
      const ratio = lengths[index] === 0 ? 0 : target / lengths[index];
      const from = points[index];
      const to = points[index + 1];
      return {
        x: from.x + (to.x - from.x) * ratio,
        y: from.y + (to.y - from.y) * ratio,
      };
    }
    target -= lengths[index];
  }
  return points[points.length - 1];
}
