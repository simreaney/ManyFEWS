import re
from time import sleep
from unittest import mock

from django.contrib.gis.geos import Polygon
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.core import mail
from django.test import TestCase, LiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)

from .alerts import TwilioAlerts
from .converters import BoundingBoxUrlParameterConverter

import logging

logger = logging.getLogger(__name__)


class TwilioAlertsTests(TestCase):
    @mock.patch("webapp.alerts.Client")
    def test_check_verification_code_requires_approved_status(self, client_cls):
        client_instance = client_cls.return_value
        verification_checks = (
            client_instance.verify.services.return_value.verification_checks
        )

        twilio_alerts = TwilioAlerts()

        # A correct code: Twilio reports "approved" - should verify.
        verification_checks.create.return_value.status = "approved"
        assert twilio_alerts.check_verification_code("+441234567890", "123456") is True

        # A wrong code: Twilio reports "pending" (not an exception, not an error) -
        # this must NOT be treated as verified, or any code would work.
        verification_checks.create.return_value.status = "pending"
        assert twilio_alerts.check_verification_code("+441234567890", "000000") is False

        # Any other non-"approved" status should also not verify.
        verification_checks.create.return_value.status = "canceled"
        assert twilio_alerts.check_verification_code("+441234567890", "123456") is False


class ConverterTestCase(TestCase):
    def test_bounding_box_url_parameter_converter(self):
        converter = BoundingBoxUrlParameterConverter()

        # Check conversion to bounding box
        # Simple ints
        bb = converter.to_python("1,2,3,4")
        assert bb == Polygon.from_bbox((1, 2, 3, 4))

        # Negative numbers and decimals
        bb = converter.to_python("-1,2.345,-3.456789,4")
        assert bb == Polygon.from_bbox((-1, 2.345, -3.456789, 4))

        # Invalid types
        self.assertRaises(ValueError, converter.to_python, "-1,2.345,-3.456789,4,5,6,7")
        self.assertRaises(ValueError, converter.to_python, "not,a,valid,coordinate")

        # Check regex works as expected
        assert re.fullmatch(converter.regex, "1,2,3,4")
        assert re.fullmatch(converter.regex, "-1,2.345,-3.456789,4")
        assert not re.fullmatch(converter.regex, "1,2")
        assert not re.fullmatch(converter.regex, "1,2-")
        assert not re.fullmatch(converter.regex, "not,a,valid,coordinate")


class WebAppTestCase(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Check for ChromeDriver in path and fall back to Firefox if not found
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            cls.selenium = webdriver.Chrome(options=options)
            cls.selenium.implicitly_wait(10)
            return
        except (FileNotFoundError, WebDriverException) as e:
            logger.warning("Chrome driver not found. Falling back to Firefox")
            logger.debug(e)

        options = webdriver.FirefoxOptions()
        cls.selenium = webdriver.Firefox(options=options)
        cls.selenium.implicitly_wait(10)

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super().tearDownClass()

    def assertOnPage(self, path):
        """
        Wait for the browser to land on `path`, then assert that it did.

        WebElement.click() does not reliably block until the next page has
        loaded, so reading current_url straight after a click is a race. That
        race is what made these tests fail intermittently, and at a different
        assertion each run.
        """
        expected = "%s%s" % (self.live_server_url, path)
        WebDriverWait(self.selenium, 10).until(EC.url_to_be(expected))
        assert self.selenium.current_url == expected

    def wait_for_element(self, by, value):
        """
        Wait for an element to appear and return it.

        On timeout, report the URL the browser actually ended up on and the
        start of the page source. A bare NoSuchElementException says only that
        the element is missing, not which page was being looked at - which is
        useless when the browser has landed somewhere unexpected.
        """
        try:
            return WebDriverWait(self.selenium, 10).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            raise AssertionError(
                f"{value!r} never appeared.\n"
                f"  current_url: {self.selenium.current_url}\n"
                f"  title: {self.selenium.title}\n"
                f"  page source (first 3000 chars):\n{self.selenium.page_source[:3000]}"
            )

    def header_buttons(self):
        """Re-read the top-right header buttons, which change on login/logout."""
        return self.selenium.find_elements(By.CSS_SELECTOR, ".header .col-4 .btn")

    def alert_table_rows(self, expected_count):
        """
        Return the rows of the alerts table once there are `expected_count` of
        them, retrying while the table is being re-rendered.

        Looking up the table and then its rows is two round trips, and the page
        can reload in between - which raised StaleElementReferenceException.
        """

        def _rows(driver):
            try:
                table = driver.find_element(By.TAG_NAME, "table")
                rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            except StaleElementReferenceException:
                return False
            return rows if len(rows) == expected_count else False

        return WebDriverWait(self.selenium, 10).until(_rows)

    def test_home(self):
        # Simple test that the home page loads without errors
        self.selenium.get("%s%s" % (self.live_server_url, "/"))

        # Should be 10 daily risk boxes
        daily_risk_elements = self.selenium.find_elements(By.CLASS_NAME, "daily-risk")
        assert len(daily_risk_elements) == 10
        # Each should contain 4 hourly risk boxes
        for el in daily_risk_elements:
            hourly_risk_elements = el.find_elements(By.CLASS_NAME, "risk")
            assert len(hourly_risk_elements) == 4

        # Click on an hourly risk to ensure no errors on loading new map layers
        daily_risk_elements[1].find_element(By.CLASS_NAME, "risk").click()
        sleep(1)

        # get_log is a webkit-only non-standard WebDriver extension!
        # https://github.com/mozilla/geckodriver/issues/330
        if isinstance(self.selenium, webdriver.Chrome):
            log = self.selenium.get_log("browser")
            assert len(log) == 0, "Errors in browser log:\n" + "\n".join(
                [f"{line['level']}: {line['message']}" for line in log]
            )

    def test_users(self):
        # Add a user, log in, log out
        self.selenium.get("%s%s" % (self.live_server_url, "/accounts/signup/"))

        # Check top-right buttons are "Login" and "Sign up"
        buttons = self.header_buttons()
        assert len(buttons) == 2
        assert buttons[0].text == "Login"
        assert buttons[1].text == "Sign Up"

        # Create a new user
        self.selenium.find_element(By.ID, "id_email").send_keys(
            "manyfews@mailinator.com"
        )
        self.selenium.find_element(By.ID, "id_password1").send_keys("al25ns5235")
        self.selenium.find_element(By.ID, "id_password2").send_keys("al25ns5235")
        self.selenium.find_element(By.ID, "id_agree_privacy").click()
        self.selenium.find_element(By.ID, "signup-submit").click()

        # Should be redirected to login page
        self.assertOnPage("/accounts/login/")

        # Log in
        self.selenium.find_element(By.ID, "id_username").send_keys(
            "manyfews@mailinator.com"
        )
        self.selenium.find_element(By.ID, "id_password").send_keys("al25ns5235")
        self.selenium.find_element(By.ID, "login-submit").click()

        # Should be redirected to homepage; top-right links should have changed
        self.assertOnPage("/")
        buttons = self.header_buttons()
        assert len(buttons) == 2
        assert buttons[0].text == "Alerts"
        assert buttons[1].text == "Log Out"

        # Log out
        buttons[1].click()
        # Should be redirected to homepage; top-right links should be "Login"/"Sign Up" again
        self.assertOnPage("/")
        buttons = self.header_buttons()
        assert len(buttons) == 2
        assert buttons[0].text == "Login"
        assert buttons[1].text == "Sign Up"

        # Navigate to login page and find 'reset password' button
        buttons[0].click()
        self.assertOnPage("/accounts/login/")
        reset_button = self.selenium.find_element(By.CLASS_NAME, "btn-secondary")
        reset_button.click()

        self.assertOnPage("/accounts/password_reset/")
        self.selenium.find_element(By.ID, "id_email").send_keys(
            "manyfews@mailinator.com"
        )
        self.selenium.find_element(By.ID, "reset-password-submit").click()

        # Landing on the 'done' page means the request completed, so the reset
        # email has been sent by the time we read the outbox.
        self.assertOnPage("/accounts/password_reset/done/")

        # Check email outbox - should be a password reset email
        assert len(mail.outbox) == 1
        msg = mail.outbox[0].body
        assert (
            "You're receiving this email because you requested a password reset for your user account"
            in msg
        )
        lines = msg.split("\n\n")
        reset_link = lines[2]
        assert re.match(r"http://localhost:\d+/accounts/reset/.+", reset_link)

        # Go to reset link and change password
        self.selenium.get(reset_link)
        self.wait_for_element(By.ID, "id_new_password1").send_keys("23sj4bds32")
        self.selenium.find_element(By.ID, "id_new_password2").send_keys("23sj4bds32")
        self.selenium.find_element(By.ID, "confirm-reset-password-submit").click()

        # Go back to login page and log in with new password
        self.selenium.get("%s%s" % (self.live_server_url, "/accounts/login/"))
        self.wait_for_element(By.ID, "id_username").send_keys("manyfews@mailinator.com")
        self.selenium.find_element(By.ID, "id_password").send_keys("23sj4bds32")
        self.selenium.find_element(By.ID, "login-submit").click()
        self.assertOnPage("/")

    @mock.patch("webapp.forms.TwilioAlerts")
    @mock.patch("webapp.views.TwilioAlerts")
    def test_alerts(self, forms_mock, views_mock):
        # Mock instances of TwilioAlerts in forms and views
        forms_twilio_instance = forms_mock.return_value
        forms_twilio_instance.send_verification_mock.return_value = True
        views_twilio_instance = views_mock.return_value
        views_twilio_instance.check_verification_code.return_value = True

        # Create a user and log in
        self.selenium.get("%s%s" % (self.live_server_url, "/accounts/signup/"))
        self.selenium.find_element(By.ID, "id_email").send_keys(
            "manyfews@mailinator.com"
        )
        self.selenium.find_element(By.ID, "id_password1").send_keys("al25ns5235")
        self.selenium.find_element(By.ID, "id_password2").send_keys("al25ns5235")
        self.selenium.find_element(By.ID, "id_agree_privacy").click()
        self.selenium.find_element(By.ID, "signup-submit").click()
        self.selenium.find_element(By.ID, "id_username").send_keys(
            "manyfews@mailinator.com"
        )
        self.selenium.find_element(By.ID, "id_password").send_keys("al25ns5235")
        self.selenium.find_element(By.ID, "login-submit").click()

        # Find Alerts button and click
        self.assertOnPage("/")
        buttons = self.header_buttons()
        assert buttons[0].text == "Alerts"
        buttons[0].click()

        # Check we're on alerts page
        self.assertOnPage("/alerts")

        # Look for text
        assert (
            'You have no alerts set up. Click "Add New" to create one.'
            in self.selenium.page_source
        )

        # Find "Add new" button and click
        self.selenium.find_element(By.ID, "add-new").click()

        # Wait for form to appear
        WebDriverWait(self.selenium, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".collapse.show"))
        )

        # Scroll to bottom of page
        self.selenium.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        WebDriverWait(self.selenium, 10).until(
            EC.visibility_of_element_located((By.ID, "save-alert"))
        )
        sleep(1)

        # Click "Save" without populating form - should give errors
        save_btn = self.selenium.find_element(By.ID, "save-alert")
        save_btn.click()

        # Check for errors
        form = self.selenium.find_element(By.CSS_SELECTOR, "form.alert-form")
        assert "is-invalid" in form.find_element(
            By.ID, "id_new_phone_number"
        ).get_attribute("class")
        assert (
            "You must provide a new phone number or choose an existing one."
            in form.text
        )
        assert (
            "No geometry value provided. Use the toolbox to select an area on the map."
            in form.text
        )

        # Fill in form this time
        select = Select(self.selenium.find_element(By.ID, "id_alert_type"))
        select.select_by_visible_text("SMS")
        self.selenium.find_element(By.ID, "id_new_phone_number").send_keys(
            "+441234567890"
        )

        # Scroll to bottom of page
        self.selenium.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        WebDriverWait(self.selenium, 10).until(
            EC.visibility_of_element_located((By.ID, "save-alert"))
        )
        sleep(1)

        # Draw a rectangle on the map
        rectangle_tool = self.selenium.find_element(
            By.CLASS_NAME, "leaflet-draw-draw-rectangle"
        )
        rectangle_tool.click()
        ActionChains(self.selenium).move_to_element(rectangle_tool).move_by_offset(
            100, 0
        ).click_and_hold().move_by_offset(20, 20).release().perform()

        # Save
        save_alert = self.selenium.find_element(By.ID, "save-alert")
        self.selenium.execute_script("arguments[0].scrollIntoView();", save_alert)
        WebDriverWait(self.selenium, 10).until(
            EC.element_to_be_clickable((By.ID, "save-alert"))
        )
        sleep(1)
        save_alert.click()

        # Should now have a table under "Your Alerts"
        rows = self.alert_table_rows(1)
        assert rows[0].text == "SMS +441234567890 Verify View/Edit Delete"

        # Click Verify button - should load modal
        self.selenium.find_element(
            By.CSS_SELECTOR, ".btn-primary[data-bs-verify-id]"
        ).click()
        WebDriverWait(self.selenium, 10).until(
            EC.visibility_of_element_located((By.ID, "verify-resend"))
        )
        self.selenium.find_element(By.ID, "id_verification_code").send_keys("123456")
        self.selenium.find_element(By.ID, "verify-submit").click()

        # Should be back on alerts page with alert now verified
        self.assertOnPage("/alerts")
        rows = self.alert_table_rows(1)
        assert rows[0].text == "SMS +441234567890 Yes View/Edit Delete"

        # Click edit - should reload page with form pre-populated
        rows[0].find_element(By.CLASS_NAME, "btn-secondary").click()
        edit_url = "%s%s" % (
            self.live_server_url,
            "/alerts/edit/",
        )
        WebDriverWait(self.selenium, 10).until(EC.url_contains(edit_url))
        assert self.selenium.current_url.startswith(edit_url)
        type_select = Select(self.selenium.find_element(By.ID, "id_alert_type"))
        assert type_select.first_selected_option.text == "SMS"
        phone_select = Select(self.selenium.find_element(By.ID, "id_phone_number"))
        assert phone_select.first_selected_option.text == "+441234567890"

        # Click delete - should give confirmation dialog and then delete
        self.selenium.find_element(By.CLASS_NAME, "btn-danger").click()
        WebDriverWait(self.selenium, 10).until(
            EC.visibility_of_element_located((By.ID, "delete-link"))
        )
        self.selenium.find_element(By.ID, "delete-link").click()

        # Should be back on main alerts page
        self.assertOnPage("/alerts")
        # Should be no alerts set up again
        assert (
            'You have no alerts set up. Click "Add New" to create one.'
            in self.selenium.page_source
        )
