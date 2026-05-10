import { forwardRef } from "react";

export const SelectionRect = forwardRef<HTMLDivElement>((_, ref) => {
  return (
    <div className="sl-selection" ref={ref} aria-hidden="true">
      <span className="sl-selection__corner sl-selection__corner--tl" />
      <span className="sl-selection__corner sl-selection__corner--tr" />
      <span className="sl-selection__corner sl-selection__corner--bl" />
      <span className="sl-selection__corner sl-selection__corner--br" />
      <span className="sl-selection__dim" />
    </div>
  );
});
SelectionRect.displayName = "SelectionRect";
