# Memory, visibility and synchronization

Scope: A3 CAM mixed AIC/AIV kernel maintenance. API signatures must also compile
against the selected CANN headers; no unverified signature is promised here.

For every GM/window/UB/L1/L0 buffer write down element dtype, stride, valid rows,
capacity, byte extent, alignment, owner, consumers and reuse condition. Check
integer multiplication and offset+length before address generation. Workspace
capacity is not the number of valid rows. Tail copies must use the valid extent.

DataCopy/DataCopyPad move data; completion must be ordered with the correct
pipeline event before a dependent Vector, Scalar or DMA operation. Pair each
SetFlag/WaitFlag with its producer, consumer, event ID and reuse generation.
SyncAll orders participating execution; it does not by itself establish cache
coherence for arbitrary Scalar GetValue/SetValue on shared GM. For shared state
prefer the established DMA path. If scalar cache access is necessary, prove
clean/invalidate and visibility using the actual SDK and repeated device tests.

Do not Reset a TPipe or recycle a local tensor while consumers can still use
it. Global barriers inside a data-dependent branch need a proof that every
participant takes that branch identically. Ready is published only after the
data and metadata are visible; slots are reusable only after final consumption.
