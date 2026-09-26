/**
 * The five ways an element arrives — and, unchanged, the way it leaves.
 *
 * The entrance is the movement half of 「讲到才出现」. `easing.js` ramps a
 * presence gate from `false` to `true` over `EASE_SECONDS`; this module is what
 * turns that number into a transform. It used to be one line in `drawElement` —
 * `translate(0, (1 - presence) * PRESENCE_RISE)` — and the only thing the model
 * could do with it was switch it on. Now the *shape* of the arrival is a name
 * the model picks, from the closed set below.
 *
 * **A style is a shape, not a duration.** `presence` arrives already eased and
 * every function here is a pure function of it, so no style adds time to a
 * beat. That is the whole answer to the budget question, and the budget is
 * tight: a beat runs 1.5–2.3 seconds, the ease spends 0.55 of that and an
 * accent 0.9. A style with a duration of its own would be spending the beat on
 * the transition instead of on the thing being taught.
 *
 * **Every style is the identity at `presence === 1`.** Two things fall out of
 * that, and both are load-bearing. A settled frame is byte-identical to the one
 * this module replaced, which is what keeps every recorded smoke snapshot and
 * every artifact already on disk meaning what it meant. And the exit is free:
 * an element whose gate is ramping *down* runs the same function in reverse, so
 * it sinks if it rose, shrinks if it zoomed, un-wipes if it wiped. A direction
 * flag, or a second table of "exit" motions, would be a second thing to keep in
 * step with this one — and the two would drift the first time a style changed.
 */

/**
 * The names, in table order. `registry.py`'s glosses must be these, in any order.
 *
 * The same rule `EMPHASIS_NAMES` is under, and it is a whole-file rule rather
 * than a nicety: the model writes one of these words and the player resolves it,
 * so a name on one side and not the other is a beat that does nothing — with no
 * error anywhere, because the player answers a name it does not know by doing
 * nothing at all.
 */
export const ENTRANCE_NAMES = ["fade", "rise", "drop", "zoom", "wipe"];

/** The style a spec is given when it names none — the one that was here first. */
export const DEFAULT_ENTRANCE = "rise";

/**
 * How far an arriving element travels, in stage pixels. `rise` and `drop` share
 * it, so the two vertical styles are one sign apart and nothing else.
 *
 * Sixteen rather than the ten an equivalent effect in the reference project uses
 * (10px at 1280×720, which is nearer 8 at this stage's 960×600): the choice is
 * the user's, made against a live picture, and the note here is that it is a
 * taste and not a derivation. What it has to satisfy is narrow — big enough to
 * read as arriving rather than as appearing, small enough that nothing looks
 * like it flew in from off-screen.
 *
 * Whatever the number is, **it is the same for everything**, and that is why
 * the transform is applied in the funnel rather than in each drawer. Every
 * element drawn on a host — `bubble`, `verdict`, `ordinal`, an anchored `note`,
 * `vector`, `angle` — is already positioned relative to that host, so a
 * per-drawer distance would be a second place for the two to disagree, and
 * disagreeing by even a pixel reads as the annotation sliding off the thing it
 * points at.
 */
export const PRESENCE_RISE = 16;

/**
 * How small a `zoom` starts, as a fraction of the size it lands at.
 *
 * Not zero, and that is arithmetic rather than taste: a zero scale is a
 * singular transform, and everything drawn through it collapses to a point —
 * a shape that reaches the screen as nothing and cannot be told apart from an
 * element that failed to draw. 0.86 is the taste, in the same family as
 * `PRESENCE_RISE`: enough to read as growth, not so much that the shape looks
 * like it came from somewhere else.
 */
export const PRESENCE_ZOOM = 0.86;

/**
 * The kinds whose `x`/`y` is a *host's* centre, and whose own ink is somewhere
 * else — `dx`/`dy` away from it.
 *
 * Four kinds, and the reason this is a list rather than `element.dx ?? 0` is
 * `vector`: a `VectorElement`'s `dx`/`dy` are the *arrow's own extent*, so
 * reading them as an offset would put a rotating arrow's hinge at its own tip.
 * The four below mean "the ink is not at the origin", and each says so in its
 * own docstring.
 *
 * Everything else draws about its own origin: a vector at its tail, an axis at
 * its base, an emitter at its apex, a card, a block or a bubble centred on the
 * point it was given. So no second table is needed to answer "where is this
 * element's middle" — `(0, 0)` is the answer for everything not named here, and
 * it is a fact about the drawing rather than a gap.
 */
export const OFFSET_KINDS = new Set(["bubble", "verdict", "ordinal", "note"]);

/**
 * Where an element's own ink is centred, in stage coordinates.
 *
 * `view.live` first, because an element that is moving arrives wherever it has
 * got to: a car entering on beat 3 zooms in around the car, not around the spot
 * the scene started it at. `view.live` holds a node for every element, not only
 * the ones that move, so this is the same question for all of them.
 */
export function inkCentre(element, view) {
  const node = view.live?.get(element.id) ?? element;
  if (!OFFSET_KINDS.has(element.kind)) return { x: node.x, y: node.y };
  return { x: node.x + (element.dx ?? 0), y: node.y + (element.dy ?? 0) };
}

/**
 * The point an arriving element grows about — its ink centre, or its hinge if
 * it has one.
 *
 * `BodyElement.pivot` is the exception the other seventeen kinds do not have,
 * and it is the reason this is a function rather than `inkCentre`. An `arm`
 * hinges at its shoulder, so a `zoom` that grew it about its own middle would
 * slide the shoulder sideways while it grew — the one end of it that is
 * supposed to be bolted down.
 */
export function hingePoint(element, view) {
  const centre = inkCentre(element, view);
  const [px, py] = element.pivot ?? [0, 0];
  return { x: centre.x + px, y: centre.y + py };
}

/**
 * The box a `wipe` reveals, or `null` for "this element has no box to wipe".
 *
 * `wipe` is the one style that needs to know how big the thing is, and the
 * answer is not available for every kind — which is why the vocabulary says so.
 * The box kinds carry `width`/`height`; the badge kinds carry a `size` and are
 * square; everything else is a line, an arrow or a fan, where a left-to-right
 * reveal is not a motion that would mean anything anyway.
 *
 * `traveler` is refused by name, and it is the one case worth stating: it does
 * carry a `size`, but its position is computed inside `drawTraveler` from its
 * link and its progress, so `element.x`/`y` is nowhere near where it draws. A
 * clip rect built from those numbers would reveal a token somewhere it is not.
 */
export function elementExtent(element) {
  if (element.kind === "traveler") return null;
  const width = Number(element.width ?? element.size ?? 0);
  const height = Number(element.height ?? element.size ?? 0);
  if (!(width > 0 && height > 0)) return null;
  return { width, height };
}

/**
 * What `style` does to an element that is `presence` of the way in.
 *
 * Returns an offset, a scale and a clip region — whichever of them the style
 * has an opinion about. `drawElement` applies all three and emits nothing at
 * all for the empty object `fade` returns, which is what keeps an element that
 * never hides on exactly the path it was on before this module existed.
 *
 * A name from outside this table — a spec written by hand, or by a newer
 * contract — arrives the way the default does rather than not at all, which is
 * the same fallback `emphasisAt` and `toneColor` make and for the same reason:
 * the player is handed files it did not write, and a missing picture is worse
 * than a plain one.
 */
export function entranceAt(style, presence, element, view) {
  const away = 1 - presence;
  const clipOf = (extent) => {
    const centre = inkCentre(element, view);
    return {
      x: centre.x - extent.width / 2,
      y: centre.y - extent.height / 2,
      width: extent.width * presence,
      height: extent.height,
    };
  };

  let motion;
  switch (style) {
    case "fade":
      motion = {};
      break;
    case "drop":
      motion = { dy: -away * PRESENCE_RISE };
      break;
    case "zoom":
      motion = { scale: PRESENCE_ZOOM + (1 - PRESENCE_ZOOM) * presence };
      break;
    case "wipe": {
      const extent = elementExtent(element);
      // No box, so no reveal: it rises instead, and the vocabulary's gloss for
      // `wipe` says which kinds that happens to rather than leaving the model
      // to find out from the picture.
      motion = extent === null ? { dy: away * PRESENCE_RISE } : { clip: clipOf(extent) };
      break;
    }
    // `rise` is named as well as defaulted, and the two are one branch on
    // purpose: an *unreadable* name has to draw something, and the something it
    // draws should be the same thing a spec that named nothing gets. Naming it
    // is also what lets a test walk the vocabulary against this switch — a
    // `default` that quietly covered a name the table offers would be a name
    // the model can write and the picture ignores.
    case "rise":
    default:
      motion = { dy: away * PRESENCE_RISE };
  }

  // The identity is `{}`, and it is *returned* rather than approximated. A
  // style that reported `{ dy: 0 }` or `{ scale: 1 }` would still be applied —
  // three drawing calls that cancel exactly are three drawing calls — and the
  // promise above is that a settled frame is drawn on precisely the path it was
  // on before this module existed. A full-box clip is the same argument: it
  // hides nothing and costs a `beginPath`/`rect`/`clip` to say so.
  const out = {};
  if (motion.dy) out.dy = motion.dy;
  if (motion.scale !== undefined && motion.scale !== 1) out.scale = motion.scale;
  if (motion.clip && presence < 1) out.clip = motion.clip;
  return out;
}
