import type { SupportedJobType } from "../../shared/jobTypes";
import { SCENARIOS } from "../jobRegistry";

interface ScenarioSelectionProps {
  selected: SupportedJobType | null;
  onSelect: (jobType: SupportedJobType) => void;
}

export function ScenarioSelection({ selected, onSelect }: ScenarioSelectionProps) {
  return (
    <section className={`scenario-grid ${selected ? "scenario-grid--has-selection" : ""}`}>
      {SCENARIOS.map((scenario) => {
        const isSelected = scenario.jobType === selected;
        const isCollapsed = selected !== null && !isSelected;

        return (
          <button
            type="button"
            key={scenario.jobType}
            className={`scenario-card ${isSelected ? "scenario-card--selected" : ""} ${isCollapsed ? "scenario-card--collapsed" : ""}`}
            onClick={() => onSelect(scenario.jobType)}
            aria-pressed={isSelected}
          >
            <h2>{scenario.title}</h2>
            <p>{scenario.description}</p>
          </button>
        );
      })}
    </section>
  );
}
