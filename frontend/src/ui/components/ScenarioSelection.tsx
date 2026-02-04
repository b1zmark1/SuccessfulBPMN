import type { ReactNode } from "react";
import type { SupportedJobType } from "../../shared/jobTypes";
import { SCENARIOS } from "../jobRegistry";

interface ScenarioSelectionProps {
  selected: SupportedJobType | null;
  onSelect: (jobType: SupportedJobType) => void;
  children?: ReactNode;
}

export function ScenarioSelection({ selected, onSelect, children }: ScenarioSelectionProps) {
  const selectedClass = selected ? `scenario-grid--selected-${selected}` : "";

  return (
    <section className={`scenario-grid ${selected ? "scenario-grid--has-selection" : ""} ${selectedClass}`.trim()}>
      {SCENARIOS.map((scenario) => {
        const isSelected = scenario.jobType === selected;
        const isCollapsed = selected !== null && !isSelected;

        return (
          <article
            key={scenario.jobType}
            className={`scenario-card ${isSelected ? "scenario-card--selected" : ""} ${isCollapsed ? "scenario-card--collapsed" : ""}`}
          >
            <button
              type="button"
              className="scenario-card__trigger"
              onClick={() => onSelect(scenario.jobType)}
              aria-pressed={isSelected}
            >
              <h2>{scenario.title}</h2>
              <p>{scenario.description}</p>
            </button>
            {isSelected && children ? <div className="scenario-card__workspace">{children}</div> : null}
          </article>
        );
      })}
    </section>
  );
}
