

import HealthIndicator from "./HealthIndicator";
import UploadControl from "./UploadControl";

export default function Header({
  healthState,
  health,
  onUpload,
  uploadLoading,
}) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark">
          W
        </div>

        <div>
          <h1>WordLens</h1>

          <p>
            Research paper terminology assistant
          </p>
        </div>
      </div>

      <div className="header-actions">
        <HealthIndicator
          state={healthState}
          health={health}
        />

        <UploadControl
          onUpload={onUpload}
          loading={uploadLoading}
          disabled={
            healthState === "checking"
          }
        />
      </div>
    </header>
  );
}