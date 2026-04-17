const GrainOverlay = () => {
  return (
    <>
      <div aria-hidden className="pointer-events-none fixed inset-0 z-[9990]">
        <svg
          className="w-full h-full opacity-[0.12] mix-blend-overlay"
          xmlns="http://www.w3.org/2000/svg"
        >
          <filter id="grainFilter">
            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch" />
            <feColorMatrix type="saturate" values="0" />
          </filter>
          <rect width="100%" height="100%" filter="url(#grainFilter)" />
        </svg>
      </div>
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[9989]"
        style={{
          background:
            'radial-gradient(1200px circle at 50% -10%, rgba(198,255,61,0.07), transparent 60%), radial-gradient(900px circle at 100% 100%, rgba(91,108,255,0.06), transparent 60%)',
        }}
      />
    </>
  );
};

export default GrainOverlay;
