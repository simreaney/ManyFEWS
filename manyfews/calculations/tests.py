from datetime import datetime, timedelta, timezone
import os, tempfile

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.gis.geos import MultiPolygon, Point, Polygon
from django.test import TestCase, override_settings
from django.urls import reverse
import numpy as np
from unittest import mock

from webapp.models import UserAlert, UserPhoneNumber, AlertType
from .alerts import send_phone_alerts_for_user
from .flood_risk import predict_depth, predict_depths
from .generate_river_flows import prepareWeatherForecastData
from .models import (
    DepthPrediction,
    FloodModelParameters,
    ModelVersion,
    RiverChannel,
    NoaaForecast,
    InitialCondition,
    AggregatedWeatherReading,
    RiverFlowPrediction,
    RiverFlowCalculationOutput,
    TestModeSettings,
)
from .tasks import (
    initialModelSetUp,
    dailyModelUpdate,
    recalculate_flood_flows,
    send_alerts,
    load_params_from_csv,
)
from .open_meteo import offsetTime


class _FakeOpenMeteoResponse:
    """A minimal stand-in for requests.Response, for mocking Open-Meteo calls."""

    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _synthetic_hourly(start_date_str, end_date_str, member_suffixes=("",)):
    """
    Build a synthetic Open-Meteo "hourly" response dict covering
    [start_date, end_date] inclusive (24 hourly points/day, UTC), for one or
    more ensemble member suffixes (e.g. ("", "_member01", ...)).
    """
    start = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    num_hours = ((end.date() - start.date()).days + 1) * 24

    times = [int((start + timedelta(hours=h)).timestamp()) for h in range(num_hours)]

    hourly = {"time": times}
    for suffix in member_suffixes:
        hourly[f"precipitation{suffix}"] = [0.1] * num_hours
        hourly[f"temperature_2m{suffix}"] = [
            20.0 + (h % 24) * 0.1 for h in range(num_hours)
        ]
        hourly[f"windspeed_10m{suffix}"] = [5.0] * num_hours
        hourly[f"winddirection_10m{suffix}"] = [180.0] * num_hours
        hourly[f"relativehumidity_2m{suffix}"] = [70.0] * num_hours

    return {"hourly": hourly}


def _mock_open_meteo_get(url, params=None, timeout=None):
    """
    Stand-in for calculations.open_meteo.requests.get, used to test the
    Open-Meteo integration without hitting the network. Returns synthetic
    hourly data spanning exactly the requested date range, so
    prepareOpenMeteoHistorical's completeness check passes and
    prepareOpenMeteo produces a predictable number of ensemble members.
    """
    start_date = params["start_date"]
    end_date = params["end_date"]

    if "archive-api" in url:
        return _FakeOpenMeteoResponse(_synthetic_hourly(start_date, end_date))

    # Ensemble forecast endpoint: return more members than
    # OPEN_METEO_ENSEMBLE_MEMBERS so the truncation logic gets exercised.
    member_suffixes = ("", "_member01", "_member02", "_member03", "_member04")
    return _FakeOpenMeteoResponse(
        _synthetic_hourly(start_date, end_date, member_suffixes=member_suffixes)
    )


class TaskTest(TestCase):
    @mock.patch(
        "calculations.open_meteo.requests.get", side_effect=_mock_open_meteo_get
    )
    def test_initial_model_setup(self, mock_get):
        """
        Test the initial Model SetUp and daily update tasks.
        """

        # test initial model setup task.
        initialModelSetUp()

        # Check that there are historical weather readings in the database,
        # covering INITIAL_BACKTIME days (4 six-hour buckets/day).
        location = Point(settings.LON_VALUE, settings.LAT_VALUE)
        self.timeInfo = offsetTime(backDays=settings.INITIAL_BACKTIME)
        self.startTime = self.timeInfo[0]
        self.endTime = self.timeInfo[1] + timedelta(days=settings.INITIAL_BACKTIME)

        self.aggregateReading = AggregatedWeatherReading.objects.filter(
            date__range=(self.startTime, self.endTime)
        ).filter(location=location)

        assert len(self.aggregateReading) == settings.INITIAL_BACKTIME * 4

        # check that there are initial conditions in the database
        self.initial_condition = InitialCondition.objects.all()
        assert len(self.initial_condition) == 100

        # test daily model update task.
        dailyModelUpdate()

        # Open-Meteo ensemble forecast: OPEN_METEO_ENSEMBLE_MEMBERS members,
        # each with OPEN_METEO_FORECAST_DAYS * 4 six-hour buckets.
        num_members = settings.OPEN_METEO_ENSEMBLE_MEMBERS
        buckets_per_member = settings.OPEN_METEO_FORECAST_DAYS * 4

        self.forecastReadings = NoaaForecast.objects.all()
        assert len(self.forecastReadings) == num_members * buckets_per_member

        # every member gets a saved river flow forecast (riverFlowSave=True)...
        self.riverOutput = RiverFlowCalculationOutput.objects.all()
        assert len(self.riverOutput) == num_members * buckets_per_member

        self.riverOutputPrediction = RiverFlowPrediction.objects.all()
        assert len(self.riverOutputPrediction) == num_members * buckets_per_member * 100

        # ...but only the control member's state carries forward, so only
        # 100 more InitialCondition rows get added (not num_members * 100).
        self.initialCondition = InitialCondition.objects.all()
        assert len(self.initialCondition) == 200

    def test_load_params_from_csv(self):
        self.csv = (
            "lng,lat,size,P0,P1,P2,P3,minQ\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-1.17E-05,4.56E-09,125\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,1.07E-05,-2.93E-08,125\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-2.46E-05,2.50E-08,125\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-3.71E-05,4.37E-08,100\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-4.50E-05,5.54E-08,100\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-3.29E-05,3.62E-08,100\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-2.76E-05,2.69E-08,100\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-2.15E-05,1.76E-08,100\n"
            "100.0,1.0,1.8E-05,-0.7,0.01,-4.03E-05,4.71E-08,100"
        )

        self.csv = self.csv.encode("utf-8")

        self.version_name = "1"
        self.model_version = ModelVersion(
            version_name=self.version_name, is_current=True
        )
        self.model_version.save()

        assert ModelVersion.objects.first().version_name == self.version_name

        settings.DATABASE_CHUNK_SIZE = 5

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # logging.info(f"Creating temporary file: {tmp.name}")
            try:
                tmp.write(self.csv)
            finally:
                tmp.close()
                load_params_from_csv(
                    filename=tmp.name, model_version_id=self.version_name
                )
                os.unlink(tmp.name)


class UserAlertTests(TestCase):
    def setUpAlerts(self):
        # Add some user alerts to db
        self.user = User(username="user1")
        self.user.save()
        self.phone_number1 = UserPhoneNumber(
            user=self.user, phone_number="+441234567890"
        )
        self.phone_number1.save()
        self.phone_number2 = UserPhoneNumber(
            user=self.user, phone_number="+449876543210"
        )
        self.phone_number2.save()
        self.alert1 = UserAlert(
            user=self.user,
            phone_number=self.phone_number1,
            alert_type=AlertType.SMS,
            location=Polygon.from_bbox((0, 0, 10, 10)),
        )
        self.alert1.save()
        self.alert2 = UserAlert(
            user=self.user,
            phone_number=self.phone_number1,
            alert_type=AlertType.SMS,
            location=Polygon.from_bbox((0, 10, 10, 20)),
        )
        self.alert2.save()
        self.alert3 = UserAlert(
            user=self.user,
            phone_number=self.phone_number2,
            alert_type=AlertType.SMS,
            location=Polygon.from_bbox((10, 10, 20, 20)),
        )
        self.alert3.save()

    @mock.patch("calculations.tasks.send_user_sms_alerts")
    def test_send_alerts(self, mock):
        # Call send_alerts: mock should not be called as nothing in db
        send_alerts()
        mock.assert_not_called()

        self.setUpAlerts()

        # Call send_alerts again. Should not call mock as alerts not verified.
        send_alerts()
        mock.assert_not_called()

        # Verify alerts 1 and 2
        self.alert1.verified = True
        self.alert1.save()
        self.alert2.verified = True
        self.alert2.save()

        # Call send_alerts again. Should call mock delay with user and phone number1 ids
        send_alerts()
        mock.delay.assert_called_once_with(self.user.id, self.phone_number1.id)

        mock.reset_mock()

        # Verify alert 3
        self.alert3.verified = True
        self.alert3.save()
        # Call send_alerts again. Should call mock delay twice with both phone numbers
        send_alerts()
        mock.delay.assert_has_calls(
            mock.call(self.user.id, self.phone_number1.id),
            mock.call(self.user.id, self.phone_number2.id),
        )

    @mock.patch("calculations.alerts.TwilioAlerts.send_alert_sms")
    def test_send_sms_alerts(self, sms_mock):
        self.setUpAlerts()
        # No depths in db so should not make any calls to Twilio apart from constructor
        send_phone_alerts_for_user(1, 1)
        sms_mock.assert_not_called()

        # Add an DepthPrediction in a location crossing alert2 and alert3
        model_version = ModelVersion(version_name="v1", is_current=True)
        model_version.save()
        parameters = FloodModelParameters(
            model_version=model_version,
            bounding_box=Polygon.from_bbox((9, 9, 11, 11)),
            beta0=0,
        )
        parameters.save()
        prediction = DepthPrediction(
            date=datetime.utcnow().date() + timedelta(days=1),
            parameters=parameters,
            median_depth=1,
            lower_centile=0.5,
            mid_lower_centile=0.7,
            upper_centile=1.5,
            model_version=model_version,
        )
        prediction.save()

        # Call with user 1, phone number 1
        send_phone_alerts_for_user(self.user.id, self.phone_number1.id)
        assert sms_mock.call_count == 1
        call_args = sms_mock.call_args[0]
        assert call_args[0] == "+441234567890"
        assert call_args[1].startswith("Floods up to 1.0m predicted from ")
        assert call_args[1].endswith(f"See {settings.SITE_URL} for details.")

        sms_mock.reset_mock()

        # Call with user 1, phone number 2
        send_phone_alerts_for_user(self.user.id, self.phone_number2.id)
        assert sms_mock.call_count == 1
        call_args2 = sms_mock.call_args[0]
        assert call_args2[0] == "+449876543210"
        assert call_args2[1] == call_args[1]

        sms_mock.reset_mock()

        # Add a RiverChannel which covers all of the depth prediction - should not send alert
        channel = RiverChannel(
            channel_location=MultiPolygon([Polygon.from_bbox((8, 8, 12, 12))])
        )
        channel.save()
        send_phone_alerts_for_user(self.user.id, self.phone_number2.id)
        assert sms_mock.call_count == 0

        sms_mock.reset_mock()

        # Modify river channel so it only covers part of the DepthPrediction and alert intersection - should send alert
        channel.channel_location = MultiPolygon(
            [Polygon.from_bbox((10, 10, 10.5, 10.5))]
        )
        channel.save()
        send_phone_alerts_for_user(self.user.id, self.phone_number2.id)
        assert sms_mock.call_count == 1
        call_args2 = sms_mock.call_args[0]
        assert call_args2[0] == "+449876543210"
        assert call_args2[1] == call_args[1]


class FloodCalculationTests(TestCase):
    fixtures = ["ModelVersion", "FloodModelParameters"]

    def setUp(self):
        super().setUp()
        self.test_date = datetime(2015, 10, 3, 23, 55, 59, 342380)

    def create_depth_predictions(self):
        model_version = ModelVersion.objects.first()

        depth_predictions = [
            DepthPrediction.objects.create(
                date=self.test_date,
                parameters_id=param_id,
                lower_centile=0.5,
                median_depth=1,
                mid_lower_centile=0.7,
                upper_centile=1.5,
                model_version=model_version,
            )
            for param_id in range(1, 5)
        ]

    def test_predict_depth(self):
        params = FloodModelParameters(beta0=1, beta1=2, beta2=3, beta3=4)

        # Test with all the same flows so centiles and medians will be actual value
        flows = np.array([2, 2, 2, 2])
        stats = predict_depth(flows, params)
        assert stats == (49, 49, 49, 49)

        # Test with different flows
        flows = np.array([0.1, 2, 1.5, 5])
        stats = predict_depth(flows, params)
        np.testing.assert_almost_equal(stats, (8.14, 21.95, 36.63, 424.90), 2)

        # Test values below 0 are set to 0
        params = FloodModelParameters(beta0=-1, beta1=-2, beta2=-3, beta3=-4)
        flows = np.array([0.1, 2, 1.5, 5])
        stats = predict_depth(flows, params)
        assert stats == (0, 0, 0, 0)

    @mock.patch("calculations.flood_risk.predict_depth")
    def test_bulk_predict_depths_delete(self, predict_depth):
        predict_depth.return_value = (8.14, 21.95, 36.63, -1)
        dummy_param_list = [1, 2, 3, 4]
        self.create_depth_predictions()

        self.assertEqual(DepthPrediction.objects.filter(date=self.test_date).count(), 4)
        with self.assertNumQueries(4):
            predict_depths(self.test_date, dummy_param_list, None)

        self.assertEqual(DepthPrediction.objects.filter(date=self.test_date).count(), 0)

    @mock.patch("calculations.flood_risk.predict_depth")
    def test_bulk_predict_depths_create(self, predict_depth):
        predict_depth.return_value = (8.14, 21.95, 36.63, 1)

        dummy_param_list = [1, 2, 3, 4]

        self.assertEqual(DepthPrediction.objects.filter(date=self.test_date).count(), 0)
        with self.assertNumQueries(8):
            predict_depths(self.test_date, dummy_param_list, None)

        self.assertEqual(DepthPrediction.objects.filter(date=self.test_date).count(), 4)

    @mock.patch("calculations.flood_risk.predict_depth")
    def test_bulk_predict_depths_update(self, predict_depth):
        predict_depth.return_value = (8.14, 21.95, 36.63, 1)
        self.create_depth_predictions()

        dummy_param_list = [1, 2, 3, 4]

        self.assertEqual(DepthPrediction.objects.filter(date=self.test_date).count(), 4)
        with self.assertNumQueries(8):
            predict_depths(self.test_date, dummy_param_list, None)

        self.assertEqual(DepthPrediction.objects.filter(date=self.test_date).count(), 4)


class TestModeTests(TestCase):
    """
    Tests for the "test mode" 100mm/+2-day storm injection
    (TestModeSettings, generate_river_flows._testStormOverrides) and the
    admin "Recalculate flood flows" button (tasks.recalculate_flood_flows).
    """

    def setUp(self):
        self.location = Point(settings.LON_VALUE, settings.LAT_VALUE)
        self.issue_date, _ = offsetTime(backDays=0)

        rows = []
        for day in range(4):
            for bucket in range(4):
                rows.append(
                    NoaaForecast(
                        location=self.location,
                        date=self.issue_date + timedelta(hours=day * 24 + bucket * 6),
                        issue_date=self.issue_date,
                        ensemble_member="control",
                        precipitation=1.0,
                        min_temperature=290.0,
                        max_temperature=295.0,
                        wind_u=1.0,
                        wind_v=1.0,
                        relative_humidity=70.0,
                    )
                )
        NoaaForecast.objects.bulk_create(rows)

    def _forecastPrecipByDay(self):
        data = prepareWeatherForecastData(
            predictionDate=self.issue_date,
            location=self.location,
            dataSource="forecast",
            ensemble_member="control",
        )
        return data[:, 5].reshape(-1, 4)  # 4 six-hour buckets/day

    def test_storm_not_injected_when_disabled(self):
        by_day = self._forecastPrecipByDay()
        assert (by_day == 1.0).all()

    def test_storm_injected_when_enabled(self):
        TestModeSettings.objects.create(enabled=True)
        by_day = self._forecastPrecipByDay()

        storm_day = by_day[TestModeSettings.STORM_DAYS_AHEAD]
        assert storm_day.sum() == TestModeSettings.STORM_TOTAL_MM
        assert (storm_day == TestModeSettings.STORM_TOTAL_MM / 4).all()

        # Every other day is untouched.
        other_days = np.delete(by_day, TestModeSettings.STORM_DAYS_AHEAD, axis=0)
        assert (other_days == 1.0).all()

    def test_disabling_reverts_to_real_forecast(self):
        test_mode = TestModeSettings.objects.create(enabled=True)
        by_day = self._forecastPrecipByDay()
        assert by_day[TestModeSettings.STORM_DAYS_AHEAD].sum() == (
            TestModeSettings.STORM_TOTAL_MM
        )

        test_mode.enabled = False
        test_mode.save()
        by_day = self._forecastPrecipByDay()
        assert (by_day == 1.0).all()

    def test_is_singleton(self):
        TestModeSettings.objects.create(enabled=True)
        # Saving again (e.g. a second .objects.create/save call) must reuse
        # pk=1 rather than creating a second row.
        TestModeSettings(enabled=False).save()
        assert TestModeSettings.objects.count() == 1
        assert not TestModeSettings.is_enabled()


class RecalculateFloodFlowsTests(TestCase):
    # Force a forecast horizon that reaches past TestModeSettings.STORM_DAYS_AHEAD
    # regardless of the environment's own OPEN_METEO_FORECAST_DAYS setting
    # (e.g. local dev envs may shrink it to keep test_initial_model_setup fast).
    @override_settings(OPEN_METEO_FORECAST_DAYS=4, OPEN_METEO_ENSEMBLE_MEMBERS=2)
    @mock.patch(
        "calculations.open_meteo.requests.get", side_effect=_mock_open_meteo_get
    )
    @mock.patch("calculations.tasks.run_all_flood_models")
    def test_recalculate_flood_flows(self, run_all_flood_models, mock_get):
        # Set up initial conditions and today's forecast, as the daily
        # scheduled pipeline would have already done.
        initialModelSetUp()
        dailyModelUpdate()

        today, _ = offsetTime(backDays=0)
        num_members = settings.OPEN_METEO_ENSEMBLE_MEMBERS
        buckets_per_member = settings.OPEN_METEO_FORECAST_DAYS * 4

        baseline_output_count = RiverFlowCalculationOutput.objects.filter(
            prediction_date=today
        ).count()
        assert baseline_output_count == num_members * buckets_per_member

        storm_day_start = today + timedelta(days=TestModeSettings.STORM_DAYS_AHEAD)
        storm_day_end = storm_day_start + timedelta(days=1)

        baseline_storm_rainfall = list(
            RiverFlowCalculationOutput.objects.filter(
                prediction_date=today,
                forecast_time__gte=storm_day_start,
                forecast_time__lt=storm_day_end,
            ).values_list("rain_fall", flat=True)
        )
        assert baseline_storm_rainfall  # sanity check: rows exist for that day
        assert all(rain_fall < 50 for rain_fall in baseline_storm_rainfall)

        # Turn on test mode and recalculate.
        TestModeSettings.objects.create(enabled=True)
        recalculate_flood_flows()

        run_all_flood_models.assert_called_once()

        # Recalculating must not duplicate today's river flow output rows...
        self.assertEqual(
            RiverFlowCalculationOutput.objects.filter(prediction_date=today).count(),
            baseline_output_count,
        )
        # ...or add extra initial-condition rows: it's not this task's job to
        # redefine tomorrow's saved model state (that's dailyModelUpdate's).
        self.assertEqual(InitialCondition.objects.count(), 200)

        expected_storm_rain_fall = (
            TestModeSettings.STORM_TOTAL_MM / 4 / settings.MODEL_TIMESTEP
        )
        storm_rainfall = list(
            RiverFlowCalculationOutput.objects.filter(
                prediction_date=today,
                forecast_time__gte=storm_day_start,
                forecast_time__lt=storm_day_end,
            ).values_list("rain_fall", flat=True)
        )
        for rain_fall in storm_rainfall:
            self.assertAlmostEqual(rain_fall, expected_storm_rain_fall, places=5)

        # Turning test mode back off and recalculating restores the real
        # forecast's rainfall.
        TestModeSettings.objects.filter(pk=1).update(enabled=False)
        recalculate_flood_flows()

        self.assertEqual(
            RiverFlowCalculationOutput.objects.filter(prediction_date=today).count(),
            baseline_output_count,
        )
        restored_storm_rainfall = list(
            RiverFlowCalculationOutput.objects.filter(
                prediction_date=today,
                forecast_time__gte=storm_day_start,
                forecast_time__lt=storm_day_end,
            ).values_list("rain_fall", flat=True)
        )
        self.assertCountEqual(restored_storm_rainfall, baseline_storm_rainfall)


class TestModeAdminTests(TestCase):
    """Smoke tests for the Test mode admin page (singleton toggle + button)."""

    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            username="admin", password="password", email="admin@example.com"
        )
        self.client.force_login(self.admin_user)

    def test_changelist_redirects_to_singleton(self):
        response = self.client.get(
            reverse("admin:calculations_testmodesettings_changelist")
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.url,
            reverse("admin:calculations_testmodesettings_change", args=[1]),
        )
        self.assertTrue(TestModeSettings.objects.filter(pk=1).exists())

    def test_change_page_has_recalculate_button(self):
        TestModeSettings.objects.create(enabled=False)
        response = self.client.get(
            reverse("admin:calculations_testmodesettings_change", args=[1])
        )
        self.assertContains(response, "Recalculate flood flows")
        self.assertContains(response, 'name="_recalculate_flood_flows"')

    @mock.patch("calculations.admin.recalculate_flood_flows")
    def test_recalculate_button_saves_toggle_and_triggers_task(
        self, recalculate_flood_flows
    ):
        TestModeSettings.objects.create(enabled=False)
        response = self.client.post(
            reverse("admin:calculations_testmodesettings_change", args=[1]),
            {"enabled": "on", "_recalculate_flood_flows": "Recalculate flood flows"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(TestModeSettings.objects.get(pk=1).enabled)
        recalculate_flood_flows.delay.assert_called_once()

    def test_plain_save_does_not_trigger_task(self):
        TestModeSettings.objects.create(enabled=False)
        with mock.patch("calculations.admin.recalculate_flood_flows") as task:
            response = self.client.post(
                reverse("admin:calculations_testmodesettings_change", args=[1]),
                {"enabled": "on", "_save": "Save"},
            )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(TestModeSettings.objects.get(pk=1).enabled)
        task.delay.assert_not_called()
