import Section from "./Section";
import { inputStyle, smallBtn, hint } from "./styles";

export default function KeySection({
  title,
  value,
  placeholder,
  onChange,
  revealKey,
  setRevealKey,
  where,
}: {
  title: string;
  value: string;
  placeholder: string;
  onChange: (v: string) => void;
  revealKey: boolean;
  setRevealKey: (f: (v: boolean) => boolean) => void;
  where: string;
}) {
  return (
    <Section title={title} status={value ? "SAVED" : "NOT SET"} good={!!value}>
      <div style={{ display: "flex", gap: 6 }}>
        <input
          type={revealKey ? "text" : "password"}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          style={inputStyle}
        />
        <button onClick={() => setRevealKey((v) => !v)} style={smallBtn}>
          {revealKey ? "Hide" : "Show"}
        </button>
      </div>
      <p style={hint}>
        Get one at <code>{where}</code>. Stored in the macOS Keychain — never
        written to disk in plaintext.
      </p>
    </Section>
  );
}
