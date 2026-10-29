<task>
Decide whether a blind reader recovered an intended character trait. The reader saw only the conversations that
were meant to show the trait and listed what they inferred. You compare that list with the intended trait.
</task>

<rules>
- Ordinary trait: `recovered` is true when an inferred statement rated `strong` or `moderate` expresses
  substantially the same trait: the same pattern of behavior or attitude, possibly in other words. A statement on a
  related topic that misses the core pattern does not count.
- Decoy (a false trait that a single one-off episode seems to suggest): `recovered` is true when the reader did NOT
  generalize it, i.e. no inferred statement rated `strong` or `moderate` asserts the decoy trait.
- `matched`: the inferred statement you matched (for a decoy: the one that wrongly asserts it), or "".
- `rationale`: one or two sentences.
</rules>

<output_only>Return only the JSON object.</output_only>
