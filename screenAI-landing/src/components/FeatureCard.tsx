import {
  MousePointer,
  Eye,
  Lock,
  Key,
  PenLine,
  History,
  Square,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

const ICONS: Record<string, LucideIcon> = {
  MousePointer,
  Eye,
  Lock,
  Key,
  PenLine,
  History,
};

type Props = {
  icon: string;
  index: number;
  title: string;
  body: string;
};

export function FeatureCard({ icon, index, title, body }: Props) {
  const Icon = ICONS[icon] ?? Square;
  return (
    <article className="sl-card">
      <div className="sl-card__inner">
        <div className="sl-card__shine" />
        <div className="sl-card__top">
          <span className="sl-card__icon">
            <Icon size={16} strokeWidth={1.4} />
          </span>
          <span className="sl-card__index">
            {String(index).padStart(2, "0")}
          </span>
        </div>
        <h3 className="sl-card__title">{title}</h3>
        <p className="sl-card__body">{body}</p>
        <div className="sl-card__rule" />
      </div>
    </article>
  );
}
