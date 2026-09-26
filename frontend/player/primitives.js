/**
 * Semantic primitives: the T1 kinds from `rendering/registry.py`, drawn.
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
import { captionTop } from "./caption.js";
import { EASE_SECONDS } from "./easing.js";
import { drawGlyph } from "./glyphs.js";

/** Font for a primitive's own annotations — tick values, measurements, names. */
const ANNOTATION_FONT = "13px Inter, 'Microsoft YaHei', sans-serif";

/**
 * How far the plate behind an annotation reaches either side of the words' middle.
 *
 * `layout.py` holds the same number as `LABEL_PLATE_HALF`, and a test keeps the
 * two equal: a body's name is placed by a stand-off the layout bakes, and this is
 * half of how that stand-off adds up — so if the two ever part company, a name
 * would be placed for one plate height and clamped against another.
 */
const ANNOTATION_PLATE_HALF = 8;

/**
 * How solid the wash inside a sensor's beam is, as a fraction of the element's
 * own colour.
 *
 * The beam used to be filled with a literal `rgba(66, 247, 255, 0.08)` — the
 * cyan of `neon`'s tone colour at eight percent — so a scene under a light
 * palette washed its sensor in a colour the palette did not contain, and a
 * `danger`-toned emitter glowed cyan anyway. Read as a fraction of the colour
 * the element is already drawn in, `0.08 × 1` is the same 0.08 it always was on
 * the default theme, and it is the right answer on the other five.
 *
 * Applied with `*=` rather than by assigning: `drawElement` sets `globalAlpha`
 * for the whole element on the way in, and overwriting it here would stop a beam
 * from fading in with everything else it belongs to. The shimmer below does the
 * same thing for the same reason.
 */
const BEAM_WASH_ALPHA = 0.08;

/**
 * Draw text centred on a point, in the theme's text colour.
 *
 * `fillText`, never `innerHTML`: a label here is model-authored data (decision
 * D4). The annotation primitives draw their own text because a `dimension` with
 * no number on it is not a drawing of a dimension.
 *
 * The note that used to end that sentence — that a `body`'s name is "a caption
 * about the object rather than part of its geometry", as though that settled it
 * — had it backwards. Being a caption is exactly why it belongs on screen, and
 * `drawVector`, `drawAxis`, `drawDimension` and `drawAngle` all draw theirs. A
 * `body` was the one scene object that did not, and the four identical rounded
 * rectangles of the avoidance document were the result.
 */
function annotation(ctx, view, text, x, y) {
  if (!text) return;
  ctx.save();
  ctx.font = ANNOTATION_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // A dark plate behind the glyphs, or a tick value lands on the shaft it labels.
  const width = ctx.measureText(text).width;
  // The plate is a fraction of the element's own opacity, never a fixed one:
  // `drawElement` sets `globalAlpha` for the whole element, so assigning here
  // would override it and the label would stay solid while its body faded.
  ctx.save();
  ctx.fillStyle = view.theme.background;
  ctx.globalAlpha *= 0.85;
  ctx.fillRect(
    x - width / 2 - 3,
    y - ANNOTATION_PLATE_HALF,
    width + 6,
    2 * ANNOTATION_PLATE_HALF,
  );
  ctx.restore();
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

/**
 * Highlighted elements glow and draw brighter; everything else stays quiet.
 *
 * The glow is `accent`, not `lineHot`. On `neon` the two are the same cyan, so
 * nothing about the default picture moved; on `paper` they are not, and the
 * difference is the point — a highlight is a *flag*, so it should light up in
 * the colour the theme reserves for flagging things, not in the colour a
 * hundred uninvolved outlines are already drawn in.
 */
function applyHighlight(ctx, theme, highlighted) {
  if (!highlighted) return;
  ctx.shadowColor = theme.accent;
  ctx.shadowBlur = 18;
}

export function drawBody(ctx, element, view) {
  const glyph = element.glyph ? view.glyphs?.[element.glyph] : null;
  if (element.glyph && !glyph) {
    // Not a fallback: drawing the plain shape instead would look deliberate.
    // `RenderSpec` already refuses to be constructed with a `body.glyph` the
    // spec does not carry, so reaching here means the player was handed a spec
    // that never went through the contract — an older file, or a hand-edited
    // one.
    throw new Error(
      `spec 里没有字形 \`${element.glyph}\` 的几何；body \`${element.id}\` 无法绘制`,
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
  // `emphasis` is *not* read here, and its absence is deliberate. The accent was
  // this function's private business until it became every primitive's — see
  // `applyAccent` in `registry.js`, which applies it in the funnel for exactly
  // this reason: a card that jolts on the beat that says 「这一个错了」 makes the
  // same gesture a body does, and two places doing it would be two places to
  // disagree about the pivot.
  //
  // What stays here is `scale`, which is a *state*: a beat that sets it means
  // the body is bigger from now on. The accent is a gesture that is over inside
  // its own beat, added on top and thrown away — `emphasis.js` is where that
  // line is drawn, and it is the reason nothing here writes to `view.live`.
  const size = typeof scale === "number" && scale > 0 ? scale : 1;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  // A body points somewhere, and a beat may turn it. Rotating here rather than
  // in the shape builder keeps the atoms free of state, and it is what makes the
  // baseline's `{"car": {"heading": -30}}` beat — 转向绕行 — visible at all.
  //
  // *About what* is `element.pivot`, and the two-step translate is the standard
  // way to say it: move to the pivot, turn, move back. For the `(0, 0)` every
  // role but `arm` is given, the two translates cancel exactly and this is the
  // rotation about the shape's centre it has always been. `arm` gets its bottom
  // edge instead, so `heading` swings it from the shoulder rather than spinning
  // it through its own middle.
  //
  // Baked by layout from the role, never sent in `props` — a pivot is a fact
  // about the kind of thing a body is, and letting a model write one would put a
  // coordinate back in the prompt through a side door.
  const heading = rad(view.lookup(element.id, "heading", element.heading ?? 0));
  const pivot = element.pivot ?? [0, 0];
  ctx.translate(node.x, node.y);
  ctx.translate(pivot[0], pivot[1]);
  ctx.rotate(heading);
  ctx.translate(-pivot[0], -pivot[1]);

  if (glyph) {
    // The glyph brings its own geometry and its own styling; the parametric
    // path below is what a body draws when it did not ask for one. What the two
    // still share is everything that is *about the body* rather than about its
    // outline: the highlight, the rotation, the direction triangle, the label.
    //
    // `view.time` goes in because a glyph may carry a part that turns on its own
    // — the lidar's sweep arm. It is simulation time, not wall time, so the arm
    // is at the same angle on the same frame of the same spec every run.
    drawGlyph(ctx, glyph, element.width * size, element.height * size, color, view.time);
  } else {
    if (element.shape === "circle") {
      // Inscribed in the box, not circumscribed about it: the layout reserved
      // that box, and a circle drawn from the width alone would put ink outside
      // it — invisible to `_check_fit`, which compares boxes. The layout's
      // `_drawn_size` measures the drawn shape the same way, which is what keeps
      // a pivot on the shape's real centre.
      circlePath(ctx, 0, 0, (Math.min(element.width, element.height) * size) / 2);
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
  }

  // The direction triangle, now drawn whenever the body has a width to put it
  // on: with a live `heading` it is the one part of a rectangle that shows which
  // way the car is pointing. The old `props.show_heading` gate meant no scene in
  // the repo ever showed one.
  drawHeading(ctx, element, color, size, heading);
  ctx.restore();

  // The name the storyboard gave this object. Outside the rotated frame, so the
  // caption stays upright however the body is turned, and below the shape rather
  // than across it — the body is the thing being pointed at, and a caption over
  // its middle hides it.
  //
  // This was not drawn at all. Every one of the three documents' bodies carried a
  // `label` in the spec, and the avoidance document's first beat teaches
  // 底盘 / 电机控制器 / 2D激光雷达 / 避障控制程序 while the picture showed four
  // identical rounded rectangles. The glyph work (M3-c) is the other half of that
  // gap; this half had already been paid for by layout and thrown away here.
  if (element.label) {
    // The name belongs to the body, so it moves with it — and a body may move
    // *down*: `field` centres a projectile on the ground line, and the ground
    // line is low. Layout refuses to let a body's **box** into the caption's
    // strip and checks that on the boxes, but a name hangs outside its box and
    // layout does not model annotations at all — `AXIS_LABEL_GAP` in `layout.py`
    // records the same gap being closed by hand on the axis label, one primitive
    // over. So the guarantee is kept here instead: a name never crosses where the
    // subtitle starts, and is held up by however much it would have crossed by.
    // That is 21 units at the very most, and for every scene that is not sitting
    // on the frame's floor it is nothing at all.
    const below = node.y + labelDy(element) * size;
    annotation(
      ctx,
      view,
      element.label,
      node.x,
      Math.min(below, captionTop(view.stage) - ANNOTATION_PLATE_HALF),
    );
  }
}

/**
 * How far below the body's centre its name goes.
 *
 * Baked by layout (`BodyElement.label_dy`) and read off the element rather than
 * worked out here, for `AXIS_LABEL_GAP`'s reason: the stand-off is measured
 * against what the body *draws*, and this file cannot see that — a glyph's ink
 * is fitted into its box, and a circle is inscribed in one. See `LABEL_TAIL` in
 * `layout.py`, which owns the number now.
 *
 * Multiplied by `size` at the call site rather than here, so a body a beat
 * scales up keeps its name the same distance off its edge, which is what the
 * arithmetic this replaces did.
 *
 * The fallback is exactly that arithmetic, for a spec written before the field
 * did — an older file on disk, or one a person hand-edited. It is the wrong
 * number for a body wider than it is tall, which is the entire reason the field
 * exists; drawing the name slightly low beats drawing it at `NaN`, which is what
 * `undefined * size` would give `fillText` and which fails silently.
 */
function labelDy(element) {
  if (typeof element.label_dy === "number") return element.label_dy;
  return Math.max(element.height ?? 0, element.width ?? 0) / 2 + 13;
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
  ctx.save();
  ctx.globalAlpha *= BEAM_WASH_ALPHA;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();

  // The nearest obstacle inside range, which is the ray the baseline singles out.
  const hit = nearestObstacle(node, view, radius, fov, heading);

  ctx.lineWidth = 1;
  const step = rays > 1 ? fov / (rays - 1) : 0;
  // The shimmer multiplies the element's opacity and is scoped to the loop:
  // `drawElement` sets `globalAlpha` for the whole element, so a plain
  // assignment here would drop it, and leaving the last ray's value behind
  // would carry it into the hit ray below.
  ctx.save();
  for (let index = 0; index < rays; index += 1) {
    const angle = heading - fov / 2 + step * index;
    const reach = radius;
    // A slow shimmer so the fan reads as live. Phase comes from the ray index,
    // not from a clock, so two frames of the same spec still match.
    const alpha = 0.72 + Math.sin(view.time * 8 + index) * 0.18;
    ctx.strokeStyle = color;
    ctx.globalAlpha *= Math.abs(alpha);
    ctx.beginPath();
    ctx.moveTo(node.x, node.y);
    ctx.lineTo(node.x + Math.cos(rad(angle)) * reach, node.y + Math.sin(rad(angle)) * reach);
    ctx.stroke();
  }
  ctx.restore();

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
  // Deliberately `element.align` and not `view.lookup(element.id, "align", …)`:
  // alignment is settled once when the layout places the panel, so the baked
  // value *is* the answer, and a panel that re-aligns itself between beats reads
  // as a glitch rather than as a beat. `LIVE_PROPS` in `registry.js` leaves the
  // name out to say the same thing from the other side. It went unread entirely
  // before this — the layout computed it, the schema carried it, the drawer
  // ignored it and drew everything left.
  ctx.textAlign =
    element.align === "center" || element.align === "right" ? element.align : "left";
  const textX =
    ctx.textAlign === "left"
      ? left + padding
      : ctx.textAlign === "right"
        ? left + width - padding
        : element.x;
  lines.forEach((line, index) => {
    // `fillText`, never `innerHTML` — spec text is data, and the player does not
    // hand it to anything that parses (decision D4).
    ctx.fillText(line, textX, top + padding + lineHeight * (index + 0.5));
  });
  ctx.restore();
}

/**
 * A note: the words and nothing else.
 *
 * The whole difference from `drawReadout` is what is *not* here — no
 * `roundRectPath`, no `fill`, no `stroke`. That is worth a sentence because the
 * absence is the feature: a panel in a fixed right-hand corner is the one
 * drawing in this vocabulary that is always there once a scene asks for it, so
 * the scene pays for it whether or not it has anything to say. Words with no box
 * cost the picture only the room they take.
 *
 * **Nothing here re-derives a position.** `x`/`y` is the anchor's centre for a
 * note that has one and a stage coordinate for one that does not, and `dx`/`dy`
 * is the offset the layout chose — zero in the second case. So one expression
 * covers both placements, and the attachment pass in `behaviors.js` moves the
 * anchored ones for free because it keys on `element.anchor`.
 *
 * `element.lines` is the layout's own wrap, keyed by the live string. Drawing
 * from it rather than wrapping here keeps one measurement instead of two — the
 * same argument `drawBubble` makes, and the reason a beat that rewrites the
 * text does not draw out of the room the note reserved.
 */
export function drawNote(ctx, element, view) {
  const text = String(view.lookup(element.id, "text", element.text) ?? "");
  const lines = element.lines?.[text];
  if (!lines || lines.length === 0) return;

  const tone = view.lookup(element.id, "tone", element.tone);
  const x = element.x + (element.dx ?? 0);
  const y = element.y + (element.dy ?? 0);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  ctx.fillStyle = toneColor(view.theme, tone);
  ctx.font = `${NOTE_FONT_SIZE}px Inter, 'Microsoft YaHei', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Centred on the note's own box rather than on the first line, so a note whose
  // words a beat shortened grows downwards from the same middle instead of
  // sliding up. `NoteElement.height` is the layout's reserve for the longest
  // string this note can be given, which is what the box is.
  const top = y - element.height / 2;
  lines.forEach((line, index) => {
    // `fillText`, never `innerHTML` — spec text is data (decision D4).
    ctx.fillText(line, x, top + NOTE_LINE_HEIGHT * (index + 0.5));
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

/**
 * 对错标记, 值卡片 and 气泡, in stage units.
 *
 * These are the numbers this file and `layout.py` both have to know, and they
 * are mirrored here for the same reason `PLAYER`'s constants are: the box is the
 * layout's to decide and the type inside it is the drawer's to place, so the two
 * halves of one card are measured in two languages. A test greps this block
 * against `layout.CARD_*`, `layout.VERDICT_*` and `layout.BUBBLE_*`, so the day
 * one of them moves the mismatch is a red test rather than a card whose padding
 * is a different number from the one its box was cut for.
 *
 * `BUBBLE_MAX_WIDTH`, `BUBBLE_MIN_WIDTH`, `BUBBLE_TAIL_GAP` and
 * `BUBBLE_TAIL_WIDTH` are deliberately **not** here: the box and the tail are
 * both cut in `layout.py` and arrive baked, so the drawer never needs either
 * number and there is nothing to keep in step.
 */
const VERDICT_LABEL_FONT_SIZE = 16;
const VERDICT_LABEL_GAP = 8;
const CARD_PADDING = 16;
const CARD_VALUE_FONT_SIZE = 34;
const CARD_VALUE_LINE_HEIGHT = 42;
const CARD_TAG_HEIGHT = 30;
const BUBBLE_PADDING = 14;
const BUBBLE_LINE_HEIGHT = 26;
const BUBBLE_FONT_SIZE = 18;
const BUBBLE_RADIUS = 12;
/** The outline's width. Layout does not draw it, so it is not mirrored. */
const BUBBLE_STROKE = 2;
/**
 * A note's type. Only these two are mirrored, and the difference from the bubble
 * block above is the point: a bubble has an outline and a radius and a tail, all
 * of which layout computes and the drawer draws, so none of them needs a copy
 * here. A note has no chrome at all — the words *are* the drawing — so the type
 * size and the line spacing are the whole of what the two languages share.
 */
const NOTE_FONT_SIZE = 20;
const NOTE_LINE_HEIGHT = 28;
const ORDINAL_SIZE = 32;
const ORDINAL_FONT_SIZE = 16;
/** A step dot's ring, for the same reason `BUBBLE_STROKE` is not mirrored. */
const ORDINAL_STROKE = 2;

/**
 * `mark` -> the theme slot the badge is drawn in.
 *
 * Three words, three shapes, three colours, and the shapes are the point: a tick
 * and a cross are different drawings before they are different colours, which is
 * what WCAG 2.2 SC 1.4.1 asks for. Colour alone would make 对 and 错 the same
 * picture to a viewer who cannot separate red from green — and 对错 is a case
 * where reading it backwards is worse than not reading it at all.
 *
 * An unrecognised word falls back to `warn`, never to `ok` or `bad`. `warn` is
 * the only one of the three that asserts nothing about the object, and this
 * branch is reached only by a spec written by hand: `validation.py` refuses any
 * other word upstream, while there is still a model to retry.
 */
function markColor(theme, mark) {
  switch (mark) {
    case "ok":
      return theme.green;
    case "bad":
      return theme.danger;
    default:
      return theme.yellow;
  }
}

/**
 * `type` -> the theme slot the tag is drawn in.
 *
 * Six kinds onto six distinct slots, and the bijection is the reason this table
 * exists rather than a single neutral tag colour: a lesson about 数据类型 is
 * asking the viewer to tell six things apart, and two of them sharing a colour
 * would undo the one comparison the picture makes.
 *
 * Every slot is one the palettes already define, so a tag reads on all six
 * themes rather than only on the one it was chosen against. That `array` lands
 * on `accent` is not a collision with the highlight colour — `applyHighlight`
 * draws a glow (`shadowBlur`), not a fill, so a highlighted card still reads as
 * highlighted whatever colour its tag happens to be.
 */
function typeColor(theme, valueType) {
  switch (valueType) {
    case "string":
      return theme.green;
    case "number":
      return theme.yellow;
    case "boolean":
      return theme.magenta;
    case "null":
      return theme.muted;
    case "object":
      return theme.lineHot;
    case "array":
      return theme.accent;
    default:
      return theme.muted;
  }
}

export function drawVerdict(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  const mark = view.lookup(element.id, "mark", element.mark);
  const size = element.size || 38;
  const colour = markColor(view.theme, mark);
  // `node` is the **anchor's** centre, not the badge's. The attachment pass in
  // `behaviors.js` writes the parent's `x`/`y` into every mounted child, so `x`
  // and `y` on a mounted element are its anchor's centre and the badge's own
  // place is the `dx`/`dy` the layout baked in that frame — `drawBubble`'s
  // convention, and the one this drawer used to get wrong. It drew the badge at
  // `node.x`/`node.y`, i.e. at the middle of the thing it judges; over a 64x48
  // body that was 40 pixels from where layout put it and read as "pinned to the
  // corner" anyway.
  const cx = node.x + (element.dx ?? 0);
  const cy = node.y + (element.dy ?? 0);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));

  // The disc is drawn in the mark's colour and the glyph inside it in the page's,
  // so the two are always at full contrast with each other whatever the palette.
  ctx.beginPath();
  ctx.arc(cx, cy, size / 2, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
  // A ring in the page colour, so a badge pinned over a busy glyph still reads
  // as a disc rather than as a blot on whatever it landed on.
  ctx.lineWidth = 2;
  ctx.strokeStyle = view.theme.background;
  ctx.stroke();

  // Resolved from the *live* mark, not from a name baked once: a beat may turn
  // a 存疑 into a 通过, and the two want different outlines. The table travels
  // with the element because the player has no way to derive a mark's glyph —
  // and because the spec embeds only the ones a beat can actually reach.
  const glyphName = element.glyphs?.[mark];
  const glyph = glyphName ? view.glyphs?.[glyphName] : null;
  if (glyph) {
    ctx.save();
    ctx.translate(cx, cy);
    // A little over half the disc: the check's ink is 17x12 in the icon set's
    // 24-square and the cross's is 14x14, so a box this size lands both of them
    // inside the circle with room to spare, and `drawGlyph` fits by the smaller
    // ratio so neither is stretched.
    drawGlyph(ctx, glyph, size * 0.55, size * 0.55, view.theme.background, 0);
    ctx.restore();
  }
  ctx.restore();

  const text = view.lookup(element.id, "text", element.text);
  if (!text) return;

  // Placed on the side `layout.py` picked, which is a position and therefore the
  // layout's to choose (decision D3). A faint plate under the words for the same
  // reason `annotation` draws one: a label about an object usually lands on it.
  ctx.save();
  ctx.font = `600 ${VERDICT_LABEL_FONT_SIZE}px Inter, 'Microsoft YaHei', sans-serif`;
  ctx.textBaseline = "middle";
  const half = size / 2 + VERDICT_LABEL_GAP;
  const below = element.label_side === "below";
  const right = element.label_side !== "left";
  const labelX = below ? cx : right ? cx + half : cx - half;
  const labelY = below ? cy + half + VERDICT_LABEL_FONT_SIZE : cy;
  ctx.textAlign = below ? "center" : right ? "left" : "right";
  const width = element.label_width || ctx.measureText(String(text)).width;
  ctx.save();
  ctx.fillStyle = view.theme.background;
  ctx.globalAlpha *= 0.85;
  const plateX = below ? labelX - width / 2 : right ? labelX : labelX - width;
  ctx.fillRect(plateX - 3, labelY - 9, width + 6, 18);
  ctx.restore();
  ctx.fillStyle = view.theme.text;
  ctx.fillText(String(text), labelX, labelY);
  ctx.restore();
}

/**
 * A step number: a filled disc with a digit in it, on its anchor's top-left.
 *
 * `node` is the **anchor's** centre, not the dot's — the attachment pass in
 * `behaviors.js` writes the parent's `x`/`y` into every mounted child, so the
 * dot's own place is the `dx`/`dy` the layout baked in that frame. Same
 * convention as `drawBubble`, and the one `drawVerdict` needed fixing to match.
 *
 * **`element.n` is read straight off the element, not through `view.lookup`.**
 * That is the shape of the primitive rather than a shortcut: a step number is a
 * fact about the object and no beat can change it — that is why `ordinal` is
 * the only primitive with an empty `live_props`, and why there is no entry for
 * it in `registry.js`'s `LIVE_PROPS`. Looking it up would invent a tier a beat
 * can never reach, and a test in `test_render_registry.py` fails on exactly
 * that ("被 lookup 读了，却没进 live_props").
 *
 * The digit is `fillText` and the disc is ours. Ten circled digits as glyph
 * assets would be ten frozen outline strings — `GlyphPart` carries a path and
 * no text — and a Unicode ① would put the whole drawing at the mercy of whether
 * the fallback font has that block. Digits are the safest characters there are.
 */
export function drawOrdinal(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  const size = element.size || ORDINAL_SIZE;
  const cx = node.x + (element.dx ?? 0);
  const cy = node.y + (element.dy ?? 0);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));

  // Disc in the tone's colour, digit in the page's — `drawVerdict`'s arrangement
  // and for its reason: the two are then always at full contrast with each
  // other whatever palette is loaded, and the ring keeps the disc reading as a
  // disc when it lands on a busy glyph.
  circlePath(ctx, cx, cy, size / 2);
  ctx.fillStyle = toneColor(view.theme, element.tone);
  ctx.fill();
  ctx.lineWidth = ORDINAL_STROKE;
  ctx.strokeStyle = view.theme.background;
  ctx.stroke();
  ctx.restore();

  const digit = String(element.n ?? "");
  if (!digit) return;

  ctx.save();
  ctx.fillStyle = view.theme.background;
  ctx.font = `600 ${ORDINAL_FONT_SIZE}px Inter, 'Microsoft YaHei', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // `maxWidth` at 0.6 of the disc, so a digit the fallback font draws wide is
  // condensed to fit instead of crossing the ring — `drawCard`'s trade. A
  // two-digit number is out of reach (`ORDINAL_MAX` is 9) and this is what
  // keeps that true if the cap ever moves.
  ctx.fillText(digit, cx, cy, size * 0.6);
  ctx.restore();
}

export function drawCard(ctx, element, view) {
  const width = element.width;
  const height = element.height;
  const left = element.x - width / 2;
  const top = element.y - height / 2;
  const text = String(view.lookup(element.id, "text", element.text) ?? "");
  // Both read through `lookup`, because both are live: a beat may rewrite the
  // value, and may rewrite which kind of value it is. The words and the pill for
  // that kind travel with the element in `tags` — there is nowhere here to
  // derive them, and a tag that fell back to the declared type would draw the
  // previous beat's answer under this beat's value.
  const valueType = String(view.lookup(element.id, "type", element.value_type) ?? "");
  const tag = element.tags?.[valueType] ?? null;
  const colour = typeColor(view.theme, valueType);

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  roundRectPath(ctx, element.x, element.y, width, height, 12);
  ctx.fillStyle = view.theme.panel;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = view.theme.line;
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.fillStyle = view.theme.text;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Monospace in a stack that ends in the system's own, so the figure columns of
  // two cards line up. A CJK fallback is in it because a card with no `text`
  // falls back to its `label`, which is Chinese.
  ctx.font =
    `${CARD_VALUE_FONT_SIZE}px ui-monospace, SFMono-Regular, Consolas, ` +
    `'Liberation Mono', 'Microsoft YaHei', monospace`;
  // `maxWidth` rather than an overflow: the layout caps a card's width so a long
  // stand-in value (`"aaa..."`, which the model does write for 「某个字符串」)
  // cannot turn the row into a banner. Squeezing it keeps it inside the box that
  // was cut for it, which is the failing-in-the-safe-direction half of that.
  ctx.fillText(text, element.x, top + CARD_PADDING + CARD_VALUE_LINE_HEIGHT / 2, width - CARD_PADDING * 2);
  ctx.restore();

  if (!tag) return;

  const tagWidth = Math.min(tag.width, width - CARD_PADDING * 2);
  const tagY = top + CARD_PADDING + CARD_VALUE_LINE_HEIGHT + CARD_TAG_HEIGHT / 2;
  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));
  roundRectPath(ctx, element.x, tagY, tagWidth, CARD_TAG_HEIGHT, CARD_TAG_HEIGHT / 2);
  // A wash of the type's own colour rather than a new fill, so the tag has six
  // appearances on six palettes without a seventh colour being invented. `*=`
  // and not `=`: `drawElement` set `globalAlpha` for the whole element on the
  // way in, and assigning here would stop the card fading with its scene.
  ctx.save();
  ctx.fillStyle = colour;
  ctx.globalAlpha *= 0.18;
  ctx.fill();
  ctx.restore();
  ctx.lineWidth = 1;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.globalAlpha *= 0.55;
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = colour;
  ctx.font = `600 ${CARD_TAG_HEIGHT - 12}px Inter, 'Microsoft YaHei', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(tag.text, element.x, tagY);
  ctx.restore();
}

/**
 * A callout: a rounded box of words with a tail pointing at its anchor.
 *
 * `node` is the **anchor's** position, not the bubble's. The attachment pass in
 * `behaviors.js` writes the parent's `x`/`y` into every mounted child, so an
 * element's own `x`/`y` is its anchor's centre — this is the emitter's
 * convention, and following it is what keeps the tail correct under motion. The
 * body is placed by `dx`/`dy` and the tail by `tail`, both in the anchor's frame,
 * so one assignment moves the whole callout with the thing it annotates.
 *
 * `drawVerdict` bakes an absolute corner instead and carries an `anchor` as
 * well, and the attachment pass overwrites the corner with the anchor's centre.
 * Over a 38-pixel disc nobody notices; over a box of words it would put the
 * callout on top of the thing it is pointing at.
 *
 * No presence gate. The ramp exists for things that arrive and leave — a car, a
 * beam, a trajectory — and a callout is either written or not written. That also
 * keeps `PRESENCE_PROP` and the harness's `switchedOff` as the one global set
 * they are compared as.
 */
export function drawBubble(ctx, element, view) {
  const node = view.live.get(element.id) ?? element;
  const text = String(view.lookup(element.id, "text", element.text) ?? "");
  const outline = toneColor(view.theme, view.lookup(element.id, "tone", element.tone));
  const cx = node.x + (element.dx ?? 0);
  const cy = node.y + (element.dy ?? 0);
  const width = element.width ?? 0;
  const height = element.height ?? 0;

  ctx.save();
  applyHighlight(ctx, view.theme, view.highlighted.has(element.id));

  // The tail first, and whole. The body is then filled and stroked over its base,
  // so the seam where the two shapes meet is one line rather than two half
  // strokes a pixel apart — which is the artefact anyone would notice first.
  const tail = (element.tail ?? []).map((point) => ({
    x: node.x + point.x,
    y: node.y + point.y,
  }));
  if (tail.length === 3) {
    polylinePath(ctx, tail);
    ctx.closePath();
    ctx.fillStyle = view.theme.panel;
    ctx.fill();
    ctx.lineWidth = BUBBLE_STROKE;
    ctx.strokeStyle = outline;
    ctx.stroke();
  }

  roundRectPath(ctx, cx, cy, width, height, BUBBLE_RADIUS);
  ctx.fillStyle = view.theme.panel;
  ctx.fill();
  ctx.lineWidth = BUBBLE_STROKE;
  ctx.strokeStyle = outline;
  ctx.stroke();
  ctx.restore();

  if (!text) return;

  // The lines come from the table `layout.py` wrapped, not from a fresh wrap
  // here. `text` is live and the box was cut for the *tallest* string a beat can
  // write, so wrapping in this file would be a second estimate of a measurement
  // the box already answers to — and two estimates disagree by a line, at which
  // point the third draws out of the bottom of the bubble with nothing to
  // notice. The fallback runs only on a spec that did not come through layout.
  const lines = element.lines?.[text] ?? text.split("\n");

  ctx.save();
  ctx.fillStyle = view.theme.text;
  ctx.font = `${BUBBLE_FONT_SIZE}px Inter, 'Microsoft YaHei', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Centred rather than stacked from the top: the box holds the tallest text any
  // beat can write, so a beat saying something shorter would otherwise sit at the
  // top of its own bubble with all the slack below it.
  const first = cy - ((lines.length - 1) * BUBBLE_LINE_HEIGHT) / 2;
  const inner = Math.max(width - BUBBLE_PADDING * 2, 0);
  lines.forEach((line, index) => {
    // `maxWidth`, so a line the layout's em-sum under-measured is condensed into
    // the box instead of drawn past its edge — `drawCard`'s trade, and the
    // failing-in-the-safe-direction half of having two estimators at all.
    ctx.fillText(line, cx, first + index * BUBBLE_LINE_HEIGHT, inner);
  });
  ctx.restore();
}

/**
 * 结构树 and 代码块, in stage units.
 *
 * The third block of mirrored numbers in this file, and the same trade as the
 * two above it: `layout.py` cuts the box, this file puts type inside it, and the
 * two halves of one panel are written in two languages. A test greps this block
 * against `layout.BLOCK_*`, `layout.CODE_*` and `layout.TREE_*`, so the day one
 * of them moves the mismatch is a red test rather than a listing whose rows
 * slowly walk out of the bottom of their own frame.
 */
const BLOCK_PADDING = 14;
const BLOCK_HEADER_HEIGHT = 26;
const BLOCK_HEADER_FONT_SIZE = 13;
const CODE_FONT_SIZE = 15;
const CODE_LINE_HEIGHT = 24;
const CODE_GUTTER = 34;
const TREE_FONT_SIZE = 16;
const TREE_LINE_HEIGHT = 26;
const TREE_INDENT = 22;
const TREE_MARKER = 14;
const TREE_GUIDE_GAP = 5;
const TREE_ICON_SLOT = 18;
const TREE_ICON_SIZE = 15;
const TREE_BRANCH_LABEL_GAP = 10;
const TREE_BRACE_WIDTH = 11;
const TREE_BRACE_GAP = 7;

/**
 * The monospace stack both blocks draw in, and what makes the two one piece of
 * typography rather than two.
 *
 * A CJK fallback is in the stack for the reason `drawCard`'s has one: half of
 * what a lesson puts in a tree is Chinese, and a face that has no glyph for it
 * would leave the fallback to whatever the system picks.
 */
const MONO_FONT =
  "ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', 'Microsoft YaHei', monospace";

/** One function, so the two rows of a block cannot be set in two faces. */
function blockFont(size) {
  return `${size}px ${MONO_FONT}`;
}

/**
 * `kind` -> the theme slot a span is drawn in.
 *
 * Five colours for nine kinds, and the collapse is the decision rather than a
 * shortcut. Prism's own grouping — which is what the kinds are named after —
 * puts comment in one, keyword in another, and everything declarative in a
 * third; and it deliberately leaves operators and punctuation at the body
 * colour, because a listing whose commas are a sixth hue is *harder* to read,
 * not richer. VS Code's defaults make the same call.
 *
 * The `default:` is the ordinary text colour and it is load-bearing:
 * `registry.CODE_KINDS` is a closed set that `CodeSpan.kind` is typed against,
 * so an unrecognised name here means a spec written by hand or by a future
 * version — and drawing a word in the colour of prose is what a person expects
 * of a word with nothing said about it.
 */
function codeColor(theme, kind) {
  switch (kind) {
    case "comment":
      return theme.muted;
    case "keyword":
    case "literal":
      return theme.magenta;
    case "builtin":
      return theme.accent;
    case "string":
      return theme.green;
    case "number":
      return theme.yellow;
    default:
      return theme.text;
  }
}

/**
 * `focus` -> the rows to emphasise, or null for none.
 *
 * Parsed here rather than baked as two numbers because `focus` is **live**: a
 * beat sets it as a prop and the value that arrives on any given frame is
 * whatever that beat wrote. Four characters of a fixed grammar is data, not
 * code — decision D4 forbids the player *executing* what a model wrote, not
 * reading a number out of it.
 *
 * A malformed value highlights nothing instead of throwing, which is the same
 * fallback `toneColor` and `emphasisAt` make and for the same reason:
 * `focus_not_a_range` and `focus_out_of_range` in `validation.py` refuse these
 * while there is still a model to retry, so reaching here means a hand-written
 * spec — and a listing drawn without its emphasis band beats no listing.
 */
function parseFocus(value) {
  const match = /^(\d+)(?:-(\d+))?$/.exec(String(value ?? "").trim());
  if (match === null) return null;
  const first = Number(match[1]);
  const last = match[2] === undefined ? first : Number(match[2]);
  if (first < 1 || last < first) return null;
  return { first, last };
}

/**
 * The frame both blocks are drawn in: a panel, the object's name in a strip
 * across the top, and a rule under it.
 *
 * Shared because the two primitives differ only in their row height — and
 * because a second copy of this would be a second place for the padding to drift
 * from the number `layout.py` cut the box with. The header is a *reservation*
 * rather than an overlay, the way the caption's strip is: the rows start below
 * it, so a title never lands on the first line of the thing it names.
 */
function drawBlockPanel(ctx, view, element, left, top, width, height, tone) {
  const theme = view.theme;
  ctx.save();
  applyHighlight(ctx, theme, view.highlighted.has(element.id));
  roundRectPath(ctx, element.x, element.y, width, height, 10);
  ctx.fillStyle = theme.panel;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = toneColor(theme, tone);
  ctx.stroke();
  ctx.restore();

  if (!element.label) return;

  ctx.save();
  ctx.font = `600 ${BLOCK_HEADER_FONT_SIZE}px Inter, 'Microsoft YaHei', sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = theme.muted;
  // `maxWidth` rather than an overflow, for the reason `drawCard` clamps its
  // value: a label is capped at 24 characters by the schema, which at this size
  // is wider than a block whose content is a single short word — and a title
  // that runs past its own panel is the one kind of overflow a reader would
  // blame on the picture rather than on the text.
  ctx.fillText(element.label, left + BLOCK_PADDING, top + BLOCK_HEADER_HEIGHT / 2, width - BLOCK_PADDING * 2);
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = theme.line;
  // A fraction of the element's own opacity, never a fixed one: `drawElement`
  // set `globalAlpha` on the way in, and assigning here would stop the rule
  // fading with the block it belongs to.
  ctx.globalAlpha *= 0.7;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left + 1, top + BLOCK_HEADER_HEIGHT);
  ctx.lineTo(left + width - 1, top + BLOCK_HEADER_HEIGHT);
  ctx.stroke();
  ctx.restore();
}

/**
 * The band behind the rows a beat is pointing at.
 *
 * A rectangle in the theme's accent at a fraction of its own strength, drawn
 * *behind* the text and over the panel — the move Manim's highlighted code line
 * makes with a `SurroundingRectangle`. It is what makes 「看第 3 行」 a beat
 * rather than a caption: without it the block is a still, and `focus` would be
 * one more prop the vocabulary offers and nothing reads.
 *
 * **It arrives with the beat.** `focus` is a string, and `blend` hands every
 * string straight to its target, so no amount of work in `easing.js` would ever
 * have ramped this band — the envelope below is the only place it can be given
 * a fade at all. It is on `view.time`, the same beat clock `emphasisAt` reads,
 * and it costs no new duration: by `EASE_SECONDS` the gain is exactly 1, which
 * is what leaves a settled frame looking the way it always has.
 *
 * The band spans the whole panel, line-number gutter included. That is
 * deliberate: `tree` shares this function and has no gutter, so a band that
 * stopped short of the numbers would be a second geometry for the two
 * primitives to disagree about — and a band that stops before the number it is
 * pointing at reads as though the number were an annotation to the row rather
 * than part of it.
 */
function drawFocusBand(ctx, view, left, width, top, lineHeight, focus, count) {
  if (focus === null) return;
  const first = Math.max(1, focus.first);
  const last = Math.min(count, focus.last);
  if (last < first) return;
  const gain = Math.min(1, view.time / EASE_SECONDS);
  ctx.save();
  ctx.fillStyle = view.theme.accent;
  ctx.globalAlpha *= 0.14 * gain;
  ctx.fillRect(left + 2, top + (first - 1) * lineHeight, width - 4, (last - first + 1) * lineHeight);
  ctx.restore();
}

export function drawCode(ctx, element, view) {
  const width = element.width;
  const height = element.height;
  const left = element.x - width / 2;
  const top = element.y - height / 2;
  const focus = parseFocus(view.lookup(element.id, "focus", element.focus));
  const tone = view.lookup(element.id, "tone", element.tone);

  drawBlockPanel(ctx, view, element, left, top, width, height, tone);

  const firstLine = top + BLOCK_HEADER_HEIGHT + BLOCK_PADDING;
  const codeLeft = left + BLOCK_PADDING + CODE_GUTTER;
  drawFocusBand(ctx, view, left, width, firstLine, CODE_LINE_HEIGHT, focus, element.lines.length);

  ctx.save();
  ctx.font = blockFont(CODE_FONT_SIZE);
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  element.lines.forEach((line, index) => {
    const y = firstLine + (index + 0.5) * CODE_LINE_HEIGHT;

    // The line number, right-aligned in the gutter and drawn in `muted` rather
    // than in the text colour: it is how a narration says 看第 3 行, not part of
    // the code, and a listing whose numbers read as loudly as its identifiers is
    // harder to skim than one with no numbers at all.
    ctx.fillStyle = view.theme.muted;
    ctx.textAlign = "right";
    ctx.fillText(String(index + 1), codeLeft - 10, y);

    ctx.textAlign = "left";
    let x = codeLeft;
    for (const span of line.spans) {
      ctx.fillStyle = codeColor(view.theme, span.kind);
      ctx.fillText(span.text, x, y);
      // Advanced by what the browser measured rather than by a baked column.
      // A monospace face is monospace only for the glyphs it *has*: a Chinese
      // comment falls through to a full-width fallback, and counting characters
      // would put every span after it out of step with the one before.
      x += ctx.measureText(span.text).width;
    }
  });
  ctx.restore();
}

/** The down triangle that marks a node with children, in its row's marker slot. */
function branchMarker(ctx, x, y, colour) {
  const half = 4;
  ctx.beginPath();
  ctx.moveTo(x - half, y - half * 0.75);
  ctx.lineTo(x + half, y - half * 0.75);
  ctx.lineTo(x, y + half * 0.75);
  ctx.closePath();
  ctx.fillStyle = colour;
  ctx.fill();
}

/** The small disc a node-link tree puts where a row's words begin. */
function nodeDot(ctx, x, y, colour) {
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
}

/* ------------------------------------------------------------------ 结构树 */

/**
 * Where a panel's rows live: a row index in, stage coordinates out.
 *
 * `layout.py` bakes `x`/`y` for every form — see `models.TreeElement` for why —
 * so the arithmetic here is `line.x ?? ...` and the fallback is not decoration: a
 * spec written before the field existed carries `null`, and `depth *
 * TREE_INDENT` / `index * TREE_LINE_HEIGHT` is exactly what that older player
 * drew. One expression rather than a version check, because a branch on which
 * player wrote the file is one more thing to get wrong about a file that only
 * has to keep looking the way it already looked.
 *
 * The baked numbers are offsets from the panel's text origin; the two `+`s are
 * what makes them coordinates. Returned as a closure rather than as arguments on
 * six call sites: five drawers need the same pair of origins, and threading them
 * through each of those signatures would be five chances to pass `top` where
 * `left` was meant.
 */
function rowPlacer(left, top) {
  const origin = left + BLOCK_PADDING;
  const band = top + BLOCK_HEADER_HEIGHT + BLOCK_PADDING;
  return (line, index) => ({
    x: origin + (line.x ?? line.depth * TREE_INDENT),
    y: band + (line.y ?? (index + 0.5) * TREE_LINE_HEIGHT),
  });
}

/**
 * How far a row's words start past its column: the marker slot, and the icon
 * slot when *any* row in this tree wears one.
 *
 * Asked of the whole row list rather than of the row, and it has to be: a level's
 * words start at one column whether or not the row above happened to carry a
 * mark. `layout._placed_tree` reserves the same slot by the same test, and the
 * two disagreeing would put the words of a marked tree one icon-width out of the
 * box that was cut for them.
 */
function treeLead(lines) {
  return TREE_MARKER + (lines.some((line) => line.icon) ? TREE_ICON_SLOT : 0);
}

/**
 * `spans[i]` is one past the last row of row `i`'s subtree.
 *
 * Two of the three forms are drawn per *subtree* rather than per row — a brace
 * wraps a run of siblings, and a node-link tree needs to know which rows are a
 * node's descendants. Both read it off the depths, which is why it is not
 * in the spec: it is a consequence of `depth`, not a third thing to keep in step
 * with it. The same scan runs in `layout._tree_spans`, and a test holds the two
 * implementations to the same answers on every fixture.
 */
function subtreeSpans(lines) {
  const spans = new Array(lines.length).fill(lines.length);
  const stack = [];
  lines.forEach((line, index) => {
    while (stack.length > 0 && lines[stack[stack.length - 1]].depth >= line.depth) {
      spans[stack.pop()] = index;
    }
    stack.push(index);
  });
  return spans;
}

/** One row's words, with its icon and its key/value split. */
function drawTreeRow(ctx, view, line, x, y, lead, glyphs) {
  const theme = view.theme;
  const glyph = line.icon ? glyphs[line.icon] : undefined;
  if (glyph !== undefined) {
    ctx.save();
    // Centred in the icon slot, which starts past the marker slot. `drawGlyph`
    // draws about the current origin, so the translate *is* the placement.
    ctx.translate(x + TREE_MARKER + TREE_ICON_SLOT / 2, y);
    drawGlyph(ctx, glyph, TREE_ICON_SIZE, TREE_ICON_SIZE, theme.muted, view.time);
    ctx.restore();
  }
  let cursor = x + lead;
  if (line.prefix) {
    ctx.fillStyle = theme.magenta;
    ctx.fillText(line.prefix, cursor, y);
    cursor += ctx.measureText(line.prefix).width;
  }
  ctx.fillStyle = codeColor(theme, line.value_kind);
  ctx.fillText(line.text, cursor, y);
}

/** Every row's words, drawn last so no guide or frame can land on top of them. */
function drawTreeRows(ctx, element, view, lead, at) {
  ctx.save();
  ctx.font = blockFont(TREE_FONT_SIZE);
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  element.lines.forEach((line, index) => {
    const { x, y } = at(line, index);
    drawTreeRow(ctx, view, line, x, y, lead, view.glyphs ?? {});
  });
  ctx.restore();
}

/** `outline`: a rule per ancestor level, and a triangle on rows that have children. */
function drawOutlineGuides(ctx, element, view, at) {
  const theme = view.theme;
  ctx.save();
  ctx.strokeStyle = theme.line;
  ctx.globalAlpha *= 0.6;
  ctx.lineWidth = 1;
  element.lines.forEach((line, index) => {
    const { x, y } = at(line, index);
    const half = TREE_LINE_HEIGHT / 2 + 0.5;
    for (let level = 0; level < line.depth; level += 1) {
      const guide = x - level * TREE_INDENT - TREE_GUIDE_GAP;
      ctx.beginPath();
      ctx.moveTo(guide, y - half);
      ctx.lineTo(guide, y + half);
      ctx.stroke();
    }
    // A row has children exactly when the row under it is deeper — the one
    // structural fact a drawer works out for itself, and only the outline needs
    // to: every other form says it with a brace, a frame or a connector.
    const next = element.lines[index + 1];
    if (next !== undefined && next.depth > line.depth) {
      branchMarker(ctx, x + TREE_MARKER / 2 - 3, y, theme.muted);
    }
  });
  ctx.restore();
}

/** `brace`: a curly brace around each run of siblings, in the strip reserved left. */
function drawBraces(ctx, element, view, at) {
  const theme = view.theme;
  const spans = subtreeSpans(element.lines);
  ctx.save();
  ctx.strokeStyle = theme.line;
  ctx.globalAlpha *= 0.75;
  ctx.lineWidth = 1.4;
  ctx.lineCap = "round";
  element.lines.forEach((line, index) => {
    if (spans[index] === index + 1) return; // no children, nothing to group
    const top = at(line, index).y;
    const bottom = at(element.lines[spans[index] - 1], spans[index] - 1).y;
    brace(ctx, at(line, index).x - TREE_BRACE_GAP, top, bottom);
  });
  ctx.restore();
}

/**
 * A curly brace spanning `top`..`bottom`, opening leftward from `x`.
 *
 * Four quadratics and a straight middle, which is Manim's `Brace` shape drawn as
 * a stroke instead of a filled path. The middle spike points *left*, at the row
 * whose children this is: the brace wraps a run and says what owns it, and an
 * owner that is not on the side the spike points to is a brace saying nothing.
 */
function brace(ctx, x, top, bottom) {
  const hook = Math.min(6, (bottom - top) / 3);
  const middle = (top + bottom) / 2;
  ctx.beginPath();
  ctx.moveTo(x + TREE_BRACE_WIDTH, top);
  ctx.quadraticCurveTo(x, top, x, top + hook);
  ctx.lineTo(x, middle - hook);
  ctx.quadraticCurveTo(x, middle, x - TREE_BRACE_WIDTH / 2, middle);
  ctx.quadraticCurveTo(x, middle, x, middle + hook);
  ctx.lineTo(x, bottom - hook);
  ctx.quadraticCurveTo(x, bottom, x + TREE_BRACE_WIDTH, bottom);
  ctx.stroke();
}

/**
 * `branch`: a connector from a node to each of its children.
 *
 * The same curve d3 calls `bumpX` — `bezierCurveTo(mid, y0, mid, y1, x1, y1)`,
 * one call per edge. It leaves a node horizontally, so a fan of children reads
 * as a fan rather than as a sheaf of straight lines.
 *
 * The parent's end is the *far side of its label*, measured here rather than
 * baked: the words are drawn by this file, so this is the only place that knows
 * how wide they came out, and a connector stopping short of its own label would
 * be a line through the middle of a word.
 */
function drawBranches(ctx, element, view, lead, at) {
  const theme = view.theme;
  const spans = subtreeSpans(element.lines);
  ctx.save();
  ctx.strokeStyle = theme.line;
  ctx.globalAlpha *= 0.8;
  ctx.lineWidth = 1.6;
  ctx.lineCap = "round";
  ctx.font = blockFont(TREE_FONT_SIZE);
  element.lines.forEach((line, index) => {
    if (spans[index] === index + 1) return;
    const from = at(line, index);
    const label = ctx.measureText(line.prefix + line.text).width;
    const startX = from.x + lead + label + TREE_BRANCH_LABEL_GAP;
    for (let child = index + 1; child < spans[index]; child += 1) {
      if (element.lines[child].depth !== line.depth + 1) continue;
      const to = at(element.lines[child], child);
      const mid = (startX + to.x) / 2;
      ctx.beginPath();
      ctx.moveTo(startX, from.y);
      ctx.bezierCurveTo(mid, from.y, mid, to.y, to.x, to.y);
      ctx.stroke();
    }
  });
  ctx.restore();
}

/** `branch`'s node dots, behind the words. */
function drawNodes(ctx, element, view, at) {
  const theme = view.theme;
  ctx.save();
  element.lines.forEach((line, index) => {
    const { x, y } = at(line, index);
    nodeDot(ctx, x + TREE_MARKER / 2, y, theme.muted);
  });
  ctx.restore();
}

/**
 * A nested structure, in whichever of the three forms it was laid out for.
 *
 * `element.form` is read and defaulted rather than trusted, the way every other
 * spec field is. Two absences land in the same place and both mean the outline,
 * which is `layout._form_of`'s fallback said from the other side: a file written
 * before the field existed, and a file still naming a form that has since been
 * withdrawn. Neither is an error — an outline is a true picture of the same rows,
 * and a file that used to open still opens.
 */
export function drawTree(ctx, element, view) {
  const width = element.width;
  const height = element.height;
  const left = element.x - width / 2;
  const top = element.y - height / 2;
  const focus = parseFocus(view.lookup(element.id, "focus", element.focus));
  const tone = view.lookup(element.id, "tone", element.tone);
  const form = element.form ?? "outline";
  const lead = treeLead(element.lines);
  const at = rowPlacer(left, top);

  drawBlockPanel(ctx, view, element, left, top, width, height, tone);

  const firstLine = top + BLOCK_HEADER_HEIGHT + BLOCK_PADDING;
  drawFocusBand(ctx, view, left, width, firstLine, TREE_LINE_HEIGHT, focus, element.lines.length);

  if (form === "branch") {
    drawBranches(ctx, element, view, lead, at);
    drawNodes(ctx, element, view, at);
  } else if (form === "brace") drawBraces(ctx, element, view, at);
  else drawOutlineGuides(ctx, element, view, at);

  drawTreeRows(ctx, element, view, lead, at);
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
