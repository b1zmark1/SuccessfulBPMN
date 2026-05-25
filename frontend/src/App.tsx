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
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);
  const lifecycle = useJobLifecycle();

  const canSubmit = !lifecycle.isCreating && !lifecycle.isPolling;
  const scenarioLabel = useMemo(
    () => (selectedScenario ? SCENARIO_REGISTRY[selectedScenario].title : ""),
    [selectedScenario],
  );

  const onBack = () => {
    setSelectedScenario(null);
    setUploadedImageSrc(null);
    lifecycle.reset();
  };

  const handleSubmit = async (meta: Record<string, unknown>) => {
    if (!selectedScenario) return;
    // Сохраняем src загруженной картинки, чтобы показать рядом с результатом.
    const possibleImage = meta["image_url"];
    if (typeof possibleImage === "string" && possibleImage.startsWith("data:image")) {
      setUploadedImageSrc(possibleImage);
    } else {
      setUploadedImageSrc(null);
    }
    await lifecycle.submitJob({ jobType: selectedScenario, meta });
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
              onSubmit={handleSubmit}
            />

            <JobStatusView
              jobId={lifecycle.jobId}
              job={lifecycle.job}
              isCreating={lifecycle.isCreating}
              isPolling={lifecycle.isPolling}
              requestError={lifecycle.requestError}
            />

            <JobResultView
              scenario={selectedScenario}
              job={lifecycle.job}
              uploadedImageSrc={uploadedImageSrc}
            />
          </section>
        ) : null}
      </ScenarioSelection>
    </main>
  );
}
