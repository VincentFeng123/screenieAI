export default function ProviderTile({
  label,
  sub,
  active,
  onClick,
}: {
  label: string;
  sub: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        textAlign: "left",
        padding: "10px 12px",
        borderRadius: 10,
        border: active
          ? "1.5px solid var(--primary-bg)"
          : "1px solid var(--border)",
        background: active
          ? "color-mix(in srgb, var(--primary-bg) 6%, transparent)"
          : "var(--bg)",
        color: "var(--ink)",
        cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
        {label}
      </div>
      <div
        style={{
          fontSize: 11.5,
          color: "var(--ink-3)",
          marginTop: 2,
        }}
      >
        {sub}
      </div>
    </button>
  );
}
