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
 *
 * ## Reading props through `view.lookup`, not off the element
 *
 * Every quantity a *beat* can change is read as
 * `view.lookup(element.id, prop, <the baked value>)`. Layout bakes a number for
 * each one, and that number is what the element carries; the lookup is what lets
 * a step's `object_states` — or a slider — override it at playback time.
 *
 * This is not a style choice. A beat is the only thing a step can do besides
 * highlight, and `view.lookup` is the only channel a beat has (see the tier list
 * in `player.js`). A drawer that reads `element.dx` directly is a drawer whose
 * beats cannot be seen: `drawVector` did exactly that, so the projectile
 * document's five beats about 平抛/斜抛/竖直上抛 wrote `direction: 0 / 45 / 90`
 * correctly, passed every check, and drew the same arrow five times. The rule is
 * checked in Python against `registry.py`'s `live_props`; this file is where it
 * has to be true.
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
  if (view.lookup(element.id, "visible", true) === false) return;

  const node = view.live.get(element.id) ?? element;
  // Two sources, and both are needed. `isDangerous` is what `proximity_gate`
  // decides from geometry; the prop is what a *beat* decides (the hand-written
  // baseline's 阈值判断 beat sets `car.danger`). Reading only the first is why
  // that beat was invisible; reading only the second would lose the sensor.
  const danger = view.isDangerous(element.id) || view.lookup(element.id, "danger", false) === true;
  const color = toneColor(view.theme, danger ? "danger" : element.tone);
  const scale = view.lookup(element.id, "scale", 1);
  const size = typeof scale === "number" && scale > 0 ? scale : 1;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  // A body points somewhere, and a beat may turn it. Rotating here rather than
  // in the shape builder keeps the atoms free of state, and it is what makes the
  // baseline's `{"car": {"heading": -30}}` beat — 转向绕行 — visible at all.
  const heading = rad(view.lookup(element.id, "heading", element.heading ?? 0));
  ctx.translate(node.x, node.y);
  ctx.rotate(heading);

  if (element.shape === "circle") {
    circlePath(ctx, 0, 0, (element.width * size) / 2);
  } else if (element.shape === "polygon") {
    polygonPath(
      ctx,
      0,
      0,
      (element.width * size) / 2,
      element.sides ?? 6,
      // A point-up polygon, so an octagon sits on a flat edge the way the
      // baseline's obstacles do rather than on a vertex.
      rad(-90 + 180 / (element.sides ?? 6)),
      element.inner_ratio ?? 1,
    );
  } else {
    roundRectPath(ctx, 0, 0, element.width * size, element.height * size, 8);
  }

  ctx.fillStyle = view.theme.panel;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();

  // The direction triangle, now drawn whenever the body has a width to put it
  // on: with a live `heading` it is the one part of a rectangle that shows which
  // way the car is pointing. The old `props.show_heading` gate meant no scene in
  // the repo ever showed one.
  drawHeading(ctx, element, color, size, heading);
  ctx.restore();
}

/** The little direction triangle the baseline puts on its car (`app.js:732`). */
function drawHeading(ctx, element, color, size, heading) {
  const reach = (element.width ?? 40) * size * 0.62;
  const tip = 7;
  ctx.beginPath();
  ctx.moveTo(Math.cos(heading) * reach, Math.sin(heading) * reach);
  ctx.lineTo(
    Math.cos(heading) * reach - tip * Math.cos(heading - Math.PI / 5),
    Math.sin(heading) * reach - tip * Math.sin(heading - Math.PI / 5),
  );
  ctx.lineTo(
    Math.cos(heading) * reach - tip * Math.cos(heading + Math.PI / 5),
    Math.sin(heading) * reach - tip * Math.sin(heading + Math.PI / 5),
  );
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

export function drawEmitter(ctx, element, view) {
  if (view.lookup(element.id, "enabled", true) === false) return;

  const node = view.live.get(element.id) ?? element;
  const radius = view.lookup(element.id, "radius", element.radius);
  // A mounted fan points where its body points. `BodyElement.heading` documents
  // the car and its lidar as sharing one angle, and until this read the second
  // half of that was untrue: a beat turning the car left the fan behind.
  const heading =
    element.anchor && view.live.has(element.anchor)
      ? view.lookup(element.anchor, "heading", node.heading ?? element.heading ?? 0)
      : (node.heading ?? element.heading ?? 0);
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
  if (view.lookup(element.id, "enabled", true) === false) return;

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

  // Progress is normally simulation state — computed from elapsed time rather
  // than authored — so the live node is the fallback. A beat may still pin it,
  // which is how a lesson says "and here the message is halfway".
  const driven = view.live.get(element.id)?.progress ?? element.progress ?? 0;
  const progress = view.lookup(element.id, "progress", driven);
  const point = pointAlong(path.points, progress);
  const size = element.size ?? 10;
  // The token's caption: the hand-written ROS sample walks one message through
  // 待发布 → 已发布 → 已接收, and until this was read those three beats — the
  // whole publish/subscribe lesson — drew the same token three times.
  const state = view.lookup(element.id, "state", null);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.shadowColor = view.theme.yellow;
  ctx.shadowBlur = 18;
  roundRectPath(ctx, point.x, point.y, size * 2.2, size * 1.35, 4);
  ctx.fillStyle = view.theme.yellow;
  ctx.fill();
  ctx.restore();

  if (typeof state === "string" && state) {
    annotation(ctx, view, state, point.x, point.y + size * 2.2);
  }
}

export function drawReadout(ctx, element, view) {
  const text = String(view.lookup(element.id, "text", element.text) ?? "");
  const lines = text.split("\n");
  const padding = 12;
  const lineHeight = 20;
  const longest = lines.reduce((max, line) => Math.max(max, line.length), 0);
  const width = element.width || Math.min(320, Math.max(120, longest * 13 + padding * 2));
  // The panel grows to fit a longer caption a beat wrote, rather than clipping it.
  const height = Math.max(element.height, lines.length * lineHeight + padding * 2) || 0;
  const left = element.x - width / 2;
  const top = element.y - height / 2;
  const tone = view.lookup(element.id, "tone", element.tone);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  roundRectPath(ctx, element.x, element.y, width, height, 10);
  ctx.fillStyle = view.theme.panel;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(view.theme, tone);
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
  const { dx, dy } = vectorExtent(element, view);
  const tipX = node.x + dx;
  const tipY = node.y + dy;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  arrowPath(ctx, node.x, node.y, tipX, tipY, element.head);
  ctx.stroke();
  ctx.restore();

  // Named beyond the tip: which arrow is v₀ and which is gravity is the whole
  // content of the picture, and two unlabelled arrows say nothing.
  const angle = Math.atan2(dy, dx);
  annotation(ctx, view, element.label, tipX + Math.cos(angle) * 20, tipY + Math.sin(angle) * 20);
}

/**
 * How long the arrow is and which way it points, right now.
 *
 * `direction` and `magnitude` are live, so a beat can swing or scale the arrow —
 * which is the entire content of the projectile document's first scene ("平抛 /
 * 斜抛 / 竖直上抛" *is* `direction: 0 / 45 / 90`) and was invisible for as long
 * as this function did not exist.
 *
 * The length needs `element.length_scale`, which layout bakes. It has to: layout
 * normalises length *within a scene* — the largest `magnitude` gets
 * `VECTOR_MAX_LENGTH`, the rest are proportional — because a 50 m/s velocity and
 * a 9.8 m/s² acceleration drawn to one scale would be a category error. That
 * rule lives in `layout.py` and is deliberately not restated here; the player
 * multiplies by the ratio it was handed.
 *
 * Without a usable scale the baked `dx`/`dy` stand, which is also the right
 * answer for an element whose `direction` no longer moves in this scene.
 */
function vectorExtent(element, view) {
  const magnitude = view.lookup(element.id, "magnitude", null);
  const direction = view.lookup(element.id, "direction", null);
  const scale = element.length_scale;
  if (
    typeof magnitude !== "number" ||
    typeof direction !== "number" ||
    typeof scale !== "number" ||
    !(scale > 0)
  ) {
    return { dx: element.dx, dy: element.dy };
  }
  // Physics degrees are y-up and the stage is y-down. `layout._to_canvas_angle`
  // is the one place in the project that bridges the two, and this is the same
  // negation applied to the same number.
  const angle = rad(-direction);
  const length = Math.abs(magnitude) * scale;
  return { dx: Math.cos(angle) * length, dy: Math.sin(angle) * length };
}

export function drawTrace(ctx, element, view) {
  if (view.lookup(element.id, "visible", true) === false) return;

  const { dx, dy } = drift(element, view);
  const points = element.points.map((point) => ({ x: point.x + dx, y: point.y + dy }));
  // How much of the curve is drawn. Two sources, and the smaller wins:
  //
  // - a live `length`, so "把轨迹画长一点" is a thing a beat can do;
  // - how far the body it belongs to has travelled, so a thrown ball writes its
  //   own trajectory instead of finding it already drawn.
  //
  // The second is what makes a `field` scene read as a throw rather than as a
  // finished diagram with a ball sliding along one of its lines.
  const byLength = revealFraction(view.lookup(element.id, "length", null), element);
  const anchorProgress = element.anchor
    ? view.live.get(element.anchor)?.progress
    : undefined;
  const fraction =
    typeof anchorProgress === "number"
      ? Math.min(byLength ?? 1, anchorProgress)
      : byLength;
  const drawn = fraction === null ? points : trimTo(points, fraction);
  if (drawn.length < 2) return;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 2;
  ctx.strokeStyle = toneColor(view.theme, element.tone);
  if (element.dashed) ctx.setLineDash([7, 7]);
  polylinePath(ctx, drawn);
  ctx.stroke();
  ctx.restore();
}

/**
 * The share of a trace a `length` value draws, or null for "all of it".
 *
 * Layout scales `length` against `TRACE_LENGTH_REFERENCE` and clamps it into
 * `[TRACE_SPAN_MIN, TRACE_SPAN_MAX]`; the reference is published on the element
 * as `length_reference` so this file does not restate it. A length at or above
 * the reference draws the whole curve.
 */
function revealFraction(length, element) {
  const reference = element.length_reference;
  if (typeof length !== "number" || typeof reference !== "number" || !(reference > 0)) {
    return null;
  }
  return Math.max(0, Math.min(1, length / reference));
}

/** The first `fraction` of a polyline, by arc length, as a new point list. */
function trimTo(points, fraction) {
  if (fraction >= 1) return points;
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
  let target = total * fraction;
  const kept = [points[0]];
  for (let index = 0; index < lengths.length; index += 1) {
    if (target <= lengths[index]) {
      const ratio = lengths[index] === 0 ? 0 : target / lengths[index];
      const from = points[index];
      const to = points[index + 1];
      kept.push({
        x: from.x + (to.x - from.x) * ratio,
        y: from.y + (to.y - from.y) * ratio,
      });
      return kept;
    }
    target -= lengths[index];
    kept.push(points[index + 1]);
  }
  return kept;
}

export function drawAxis(ctx, element, view) {
  const angle = rad(element.heading);
  const ticks = Math.max(1, Math.round(view.lookup(element.id, "ticks", element.ticks)));
  // `range` is the axis's span in the *document's* units. It labels the ticks
  // and does not move them: the graduations stay evenly spaced along the shaft,
  // because where a value sits between two of them is the simulation's business,
  // not the axis's. Without it an axis is an arrow with notches — the projectile
  // document draws two of those and calls them 坐标轴.
  const span = view.lookup(element.id, "range", null);
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

  ctx.beginPath();
  for (let index = 1; index <= ticks; index += 1) {
    const along = element.length * (index / ticks);
    const px = element.x + Math.cos(angle) * along;
    const py = element.y + Math.sin(angle) * along;
    ctx.moveTo(px - normalX, py - normalY);
    ctx.lineTo(px + normalX, py + normalY);
  }
  ctx.stroke();
  ctx.restore();

  if (typeof span === "number" && span > 0) {
    for (let index = 1; index <= ticks; index += 1) {
      const along = element.length * (index / ticks);
      const px = element.x + Math.cos(angle) * along;
      const py = element.y + Math.sin(angle) * along;
      annotation(ctx, view, formatTick((span * index) / ticks), px + normalX * 1.6, py + normalY * 1.6);
    }
  }

  annotation(ctx, view, element.label, tipX + Math.cos(angle) * 18, tipY + Math.sin(angle) * 18);
}

/** A tick value without a trailing `.0` and without four decimals of noise. */
function formatTick(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
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
  // The measurement's caption is live, so a beat can walk 射程 R from one value
  // to another. Falls back to the object's own label, which is where layout put
  // it — `dimension.label` is a prop precisely so this beat exists.
  const caption = view.lookup(element.id, "label", element.label);
  annotation(ctx, view, caption, midX, midY - 16);
}

export function drawAngle(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  // Physics angles are y-up and the stage is y-down; `layout.py` converts for
  // the vectors it bakes into dx/dy, and this is the same conversion applied to
  // the two directions this element carries as numbers.
  const from = rad(-element.from_degrees);
  // A beat may open or close the arc — the projectile document writes
  // `degrees: 45` on the 发射角 and expects the wedge to follow.
  const degrees = view.lookup(element.id, "degrees", element.to_degrees);
  const to = rad(-degrees);
  const radius = view.lookup(element.id, "radius", element.radius);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(view.theme, element.tone);

  ctx.beginPath();
  ctx.moveTo(node.x, node.y);
  ctx.lineTo(node.x + Math.cos(from) * radius, node.y + Math.sin(from) * radius);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(node.x, node.y, radius, from, to, to < from);
  ctx.stroke();
  ctx.restore();

  const mid = (from + to) / 2;
  annotation(
    ctx,
    view,
    element.label || `${Math.round(degrees)}°`,
    node.x + Math.cos(mid) * (radius + 18),
    node.y + Math.sin(mid) * (radius + 18),
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
