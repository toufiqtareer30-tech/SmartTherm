package local.smarttherm.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import local.smarttherm.model.AppSettings;
import local.smarttherm.model.PredictionForecast;
import local.smarttherm.model.PredictionSnapshot;
import local.smarttherm.model.ThermalAlert;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class AlertServiceTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void emitsOneAlertForAnOngoingThresholdCrossing() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        PredictionForecast forecast = new PredictionForecast(
                "CPU", 80.0, 92.0, Instant.now(), Instant.now().plusSeconds(300),
                "READY", PredictionService.MODEL, 5, "test-model-v1");
        PredictionSnapshot prediction = new PredictionSnapshot(
                "READY", PredictionService.MODEL, true, Instant.now(),
                Instant.now().plusSeconds(300), 5, List.of(forecast));

        service.evaluate(prediction, AppSettings.defaults());
        service.evaluate(prediction, AppSettings.defaults());

        assertThat(service.recent()).hasSize(1);
        assertThat(service.recent().getFirst().message()).contains("CPU temperature");
        assertThat(service.recent().getFirst().active()).isTrue();
        assertThat(temporaryDirectory.resolve("alerts.json")).exists();
    }

    @Test
    void resetsAnAlertOnlyAfterTheEstimateFallsBelowTheHysteresisBand() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        service.evaluate(prediction(92.0), AppSettings.defaults());
        service.evaluate(prediction(87.0), AppSettings.defaults());

        assertThat(service.recent().getFirst().active()).isFalse();
    }

    @Test
    void cooldownPreventsAnImmediateRepeatAfterTheAlertResets() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        service.evaluate(prediction(92.0), AppSettings.defaults());
        service.evaluate(prediction(87.0), AppSettings.defaults());
        service.evaluate(prediction(92.0), AppSettings.defaults());

        assertThat(service.recent()).hasSize(1);
        assertThat(service.recent().getFirst().active()).isFalse();
    }

    @Test
    void doesNotCreateRiskAlertsFromUntrainedPredictions() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());

        service.evaluate(new PredictionSnapshot(
                "MODEL_NOT_READY", "No model", false, Instant.now(),
                Instant.now().plusSeconds(300),
                5,
                List.of(new PredictionForecast(
                        "CPU", 80.0, 120.0, Instant.now(), Instant.now().plusSeconds(300),
                        "READY", PredictionService.MODEL, 5, "test-model-v1"))), AppSettings.defaults());

        assertThat(service.recent()).isEmpty();
    }

    @Test
    void doesNotWarnWhenForecastIsUnavailable() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        PredictionSnapshot unavailable = new PredictionSnapshot(
                "MODEL_NOT_READY", "No model", false, Instant.now(), null, 5, List.of());

        service.evaluate(unavailable, AppSettings.defaults());
        service.evaluate(unavailable, AppSettings.defaults());

        assertThat(service.recent()).isEmpty();
    }

    @Test
    void resolvesPreviouslySavedMeasuredTemperatureAlertsOnLoad() throws Exception {
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        Path alertsFile = temporaryDirectory.resolve("alerts.json");
        mapper.writeValue(alertsFile.toFile(), List.of(new ThermalAlert(
                "legacy-alert", Instant.now(), "CPU", 91.0, null, 90.0,
                "WARNING", "CPU temperature reached 91.0°C", "Check cooling", true)));
        AlertService service = new AlertService(mapper, alertsFile.toString());

        service.load();

        assertThat(service.recent()).hasSize(1);
        assertThat(service.recent().getFirst().active()).isFalse();
        assertThat(mapper.readValue(alertsFile.toFile(), ThermalAlert[].class)[0].active()).isFalse();
    }

    private static PredictionSnapshot prediction(double cpuC) {
        return new PredictionSnapshot("READY", PredictionService.MODEL, true,
                Instant.now(), Instant.now().plusSeconds(300), 5,
                List.of(new PredictionForecast(
                        "CPU", 80.0, cpuC, Instant.now(), Instant.now().plusSeconds(300),
                        "READY", PredictionService.MODEL, 5, "test-model-v1")));
    }
}
