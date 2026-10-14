

const MAX_FILE_SIZE =
  25 * 1024 * 1024;

export default function UploadControl({
  onUpload,
  loading,
  disabled,
}) {
  function handleChange(event) {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    if (
      !file.name
        .toLowerCase()
        .endsWith(".pdf")
    ) {
      onUpload({
        error:
          "Please select a PDF file.",
      });

      event.target.value = "";

      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      onUpload({
        error:
          "The selected PDF is larger than 25 MB.",
      });

      event.target.value = "";

      return;
    }

    onUpload({
      file,
    });

    event.target.value = "";
  }

  return (
    <label
      className={`upload-button ${
        disabled
          ? "disabled"
          : ""
      }`}
    >
      <input
        type="file"
        accept="application/pdf,.pdf"
        onChange={handleChange}
        disabled={
          disabled || loading
        }
      />

      {loading
        ? "Processing…"
        : "Upload research PDF"}
    </label>
  );
}