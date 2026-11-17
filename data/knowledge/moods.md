# Mood guide

Each section maps everyday feeling words to the exact mood labels used in the TuneScout catalog (data/songs.csv). Energy and valence are on a 0.0 to 1.0 scale. Valence means how bright or positive a song sounds.

## Mood: happy
Catalog mood label: happy
Similar words: cheerful, joyful, sunny, feel-good, upbeat, fun, bright, good vibes.
Typical values: valence 0.75 to 0.9, energy 0.6 to 0.85.
Nearby catalog moods: uplifting, euphoric.

## Mood: uplifting
Catalog mood label: uplifting
Similar words: hopeful, inspiring, positive, motivating, sunshine, summery, warm.
Typical values: valence 0.75 to 0.95, energy 0.5 to 0.7.
Nearby catalog moods: happy, euphoric.

## Mood: euphoric
Catalog mood label: euphoric
Similar words: ecstatic, hype, rave, party peak, on top of the world, celebratory.
Typical values: valence 0.7 to 0.9, energy 0.85 to 1.0.
Nearby catalog moods: happy, intense.

## Mood: intense
Catalog mood label: intense
Similar words: powerful, driving, pumped, adrenaline, hard-hitting, aggressive but not angry.
Typical values: energy 0.85 to 1.0, valence 0.4 to 0.8.
Nearby catalog moods: angry, confident, euphoric.

## Mood: angry
Catalog mood label: angry
Similar words: furious, rage, heavy, dark and aggressive, venting.
Typical values: energy 0.9 to 1.0, valence 0.1 to 0.3.
Nearby catalog moods: intense.

## Mood: confident
Catalog mood label: confident
Similar words: swagger, bold, boss, self-assured, strut, main character.
Typical values: energy 0.7 to 0.85, valence 0.5 to 0.7.
Nearby catalog moods: intense, happy.

## Mood: chill
Catalog mood label: chill
Similar words: mellow, laid-back, easygoing, low-key, vibey, calm, cozy.
Typical values: energy 0.25 to 0.45, valence 0.5 to 0.65.
Nearby catalog moods: relaxed, focused, peaceful.
Note: "relaxed" is a separate label; "chill" does not match it exactly.

## Mood: relaxed
Catalog mood label: relaxed
Similar words: unwinding, lazy Sunday, soothing, smooth, coffee shop, easy listening.
Typical values: energy 0.3 to 0.45, valence 0.6 to 0.8.
Nearby catalog moods: chill, peaceful, romantic.

## Mood: peaceful
Catalog mood label: peaceful
Similar words: serene, tranquil, calm, meditative, gentle, quiet, sleepy.
Typical values: energy 0.1 to 0.3, valence 0.3 to 0.6.
Nearby catalog moods: chill, relaxed, melancholic.

## Mood: focused
Catalog mood label: focused
Similar words: concentration, deep work, productive, in the zone, studying, coding.
Typical values: energy 0.35 to 0.5, valence 0.5 to 0.65.
Nearby catalog moods: chill.

## Mood: moody
Catalog mood label: moody
Similar words: brooding, atmospheric, mysterious, late night, nocturnal, cinematic.
Typical values: energy 0.6 to 0.8, valence 0.35 to 0.55.
Nearby catalog moods: melancholic, confident.

## Mood: melancholic
Catalog mood label: melancholic
Similar words: sad, wistful, bittersweet, heartbroken, rainy day, blue, introspective.
Typical values: valence 0.15 to 0.35, energy 0.2 to 0.45.
Nearby catalog moods: nostalgic, moody, peaceful.

## Mood: nostalgic
Catalog mood label: nostalgic
Similar words: throwback, sentimental, memories, homesick, old times, reminiscing.
Typical values: valence 0.5 to 0.7, energy 0.4 to 0.6.
Nearby catalog moods: melancholic, relaxed.

## Mood: romantic
Catalog mood label: romantic
Similar words: love songs, date night, sensual, tender, slow dance, intimate.
Typical values: valence 0.55 to 0.75, energy 0.4 to 0.6.
Nearby catalog moods: relaxed, chill.

## Requests that avoid a feeling
"Not sad" or "nothing depressing" means aim for valence 0.55 or higher and avoid the melancholic mood. "Not too intense" or "nothing aggressive" means aim for energy 0.6 or lower and avoid the angry and intense moods. "Not boring" means aim for energy 0.5 or higher.
