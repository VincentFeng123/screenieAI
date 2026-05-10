import React from "react";

export default function Section({
  title,
  status,
  good,
  children,
}: {
  title: string;
  status?: string;
  good?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        marginTop: 18,
        padding: 16,
        background: "var(--section-bg)",
        borderRadius: 12,
        textAlign: "left",
        color: "var(--ink)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
          {title}
        </span>
        {status && (
          <span
            style={{
              fontSize: 10.5,
              padding: "2px 8px",
              borderRadius: 999,
              background: good
                ? "var(--status-ok-bg)"
                : "var(--status-neutral-bg)",
              color: good
                ? "var(--status-ok-text)"
                : "var(--status-neutral-text)",
              fontWeight: 600,
              letterSpacing: 0.4,
            }}
          >
            {status}
          </span>
        )}
      </div>
      {children}
    </section>
  );
}
