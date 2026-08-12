from django.contrib.gis import admin
from django.forms import ModelForm, FileField
from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.urls import reverse
from leaflet.admin import LeafletGeoAdmin

from .models import (
    ModelVersion,
    RiverChannel,
    RiverFlowPrediction,
    FloodModelParameters,
    TestModeSettings,
)
from .tasks import recalculate_flood_flows


@admin.register(FloodModelParameters)
class FloodModelParametersAdmin(admin.ModelAdmin):
    list_display = ["id", "model_version_id"] + [f"beta{i}" for i in range(12)]


@admin.register(RiverFlowPrediction)
class RiverFlowPredictionAdmin(admin.ModelAdmin):
    list_display = ("prediction_index", "forecast_time", "river_flow")

    def forecast_time(self, obj):
        return obj.calculation_output.forecast_time


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):

    # form = ModelVersionForm
    list_display = ("version_name", "date_created", "is_current")

    actions = ["delete_model"]

    def get_readonly_fields(self, request, obj=None):
        # Disallow editing of param file
        if obj:  # obj is not None, so this is an edit
            return [
                "param_file",
            ]
        else:  # This is an addition
            return []

    def delete_queryset(self, request, queryset):
        queryset.delete()
        self._update_current()

    def delete_model(self, request, obj):
        obj.delete()
        self._update_current()

    def _update_current(self):
        if not ModelVersion.objects.filter(is_current=True).count():
            # Find most recent model version and make it current
            current = ModelVersion.objects.latest("date_created")
            if current:
                current.is_current = True
                current.save()


@admin.register(RiverChannel)
class RiverChannelAdmin(LeafletGeoAdmin):
    display_raw = True


@admin.register(TestModeSettings)
class TestModeSettingsAdmin(admin.ModelAdmin):
    fields = ("enabled",)

    def has_add_permission(self, request):
        # Singleton: never allow creating a second row via the admin.
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        # Singleton: skip straight to the (only) row's change page, creating
        # it on first visit.
        obj, _ = TestModeSettings.objects.get_or_create(pk=1)
        return redirect(
            reverse("admin:calculations_testmodesettings_change", args=[obj.pk])
        )

    def response_change(self, request, obj):
        if "_recalculate_flood_flows" in request.POST:
            recalculate_flood_flows.delay()
            self.message_user(
                request,
                "Test mode setting saved. Recalculating flood flows in the "
                "background - this can take a while for a large model; check "
                "the Celery worker logs or Task results for progress.",
            )
            return HttpResponseRedirect(request.path)
        return super().response_change(request, obj)
