import { useMemo, useState } from "react";
import logo from "../logo.png";
import { useJobLifecycle } from "./state/useJobLifecycle";
import type { SupportedJobType } from "./shared/jobTypes";
import { SCENARIO_REGISTRY } from "./ui/jobRegistry";
import { ScenarioSelection } from "./ui/components/ScenarioSelection";
import { ScenarioInputForm } from "./ui/components/ScenarioInputForm";
import { JobStatusView } from "./ui/components/JobStatusView";
import { JobResultView } from "./ui/components/JobResultView";

export function App() {
  const [selectedScenario, setSelectedScenario] = useState<SupportedJobType | null>(null);
  const lifecycle = useJobLifecycle();

  const canSubmit = !lifecycle.isCreating && !lifecycle.isPolling;
  const scenarioLabel = useMemo(
    () => (selectedScenario ? SCENARIO_REGISTRY[selectedScenario].title : ""),
    [selectedScenario],
  );

  const onBack = () => {
    setSelectedScenario(null);
    lifecycle.reset();
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <img className="app-header__logo" src={logo} alt="SuccessfulBPMN logo" />
          <h1 className="app-header__title">
            SUCCESSFUL<span className="app-header__brand-accent">BPMN</span>
          </h1>
        </div>
      </header>

      <ScenarioSelection selected={selectedScenario} onSelect={setSelectedScenario}>
        {selectedScenario ? (
          <section className="workspace">
            <div className="workspace__header">
              <button type="button" className="back-button" onClick={onBack}>
                Назад
              </button>
              <h2>{scenarioLabel}</h2>
            </div>

            <ScenarioInputForm
              scenario={selectedScenario}
              disabled={!canSubmit}
              onSubmit={(meta) => lifecycle.submitJob({ jobType: selectedScenario, meta })}
            />

            <JobStatusView
              jobId={lifecycle.jobId}
              job={lifecycle.job}
              isCreating={lifecycle.isCreating}
              isPolling={lifecycle.isPolling}
              requestError={lifecycle.requestError}
            />

            <JobResultView scenario={selectedScenario} job={lifecycle.job} />
          </section>
        ) : null}
      </ScenarioSelection>
    </main>
  );
}
