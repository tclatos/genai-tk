"""Unit tests for genai_tk.tools.browser.auth_handlers."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from genai_tk.tools.browser import auth_handlers
from genai_tk.tools.browser.models import AuthConfig, AuthCredentials, AuthSelectors, CredentialRef


class _FakeLocator:
    def __init__(
        self,
        *,
        visible: bool = True,
        disabled: bool = False,
        disabled_sequence: list[bool] | None = None,
        on_click: Callable[[], None] | None = None,
    ) -> None:
        self.visible = visible
        self.disabled = disabled
        self._disabled_sequence = list(disabled_sequence or [])
        self.on_click = on_click
        self.click_count = 0
        self.typed_text = ""
        self.pressed_keys: list[str] = []

    async def wait_for(self, *, state: str = "visible", timeout: int | None = None) -> None:
        del timeout
        if state == "visible" and not self.visible:
            raise TimeoutError("Locator is not visible")

    async def click(self) -> None:
        self.click_count += 1
        if self.on_click is not None:
            self.on_click()

    async def type(self, char: str, delay: int = 0) -> None:
        del delay
        self.typed_text += char

    async def is_visible(self) -> bool:
        return self.visible

    async def is_disabled(self) -> bool:
        if self._disabled_sequence:
            self.disabled = self._disabled_sequence.pop(0)
        return self.disabled

    async def get_attribute(self, name: str) -> str | None:
        if name == "disabled":
            return "" if self.disabled else None
        if name == "aria-disabled":
            return "true" if self.disabled else "false"
        return None

    async def press(self, key: str) -> None:
        self.pressed_keys.append(key)


class _FakeLocatorResult:
    def __init__(self, locator: _FakeLocator) -> None:
        self.first = locator


class _FakePage:
    def __init__(
        self,
        mapping: dict[str, _FakeLocator],
        *,
        url: str = "https://example.com",
        wait_for_url_exc: Exception | None = None,
    ) -> None:
        self._mapping = mapping
        self.url = url
        self.wait_for_url_exc = wait_for_url_exc
        self.wait_for_url_calls: list[tuple[str, int]] = []
        self.wait_for_load_state_calls: list[str] = []

    def locator(self, selector: str) -> _FakeLocatorResult:
        return _FakeLocatorResult(self._mapping[selector])

    async def wait_for_url(self, pattern: str, timeout: int) -> None:
        self.wait_for_url_calls.append((pattern, timeout))
        if self.wait_for_url_exc is not None:
            raise self.wait_for_url_exc

    async def wait_for_load_state(self, state: str) -> None:
        self.wait_for_load_state_calls.append(state)


def _make_auth_config() -> AuthConfig:
    return AuthConfig(
        credentials=AuthCredentials(
            username=CredentialRef(env="TEST_BROWSER_AUTH_USERNAME"),
            password=CredentialRef(env="TEST_BROWSER_AUTH_PASSWORD"),
        ),
        selectors=AuthSelectors(
            username_input="#username",
            password_input="#password",
            submit_button="#submit",
        ),
    )


@pytest.mark.asyncio
async def test_fill_form_single_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_BROWSER_AUTH_USERNAME", "alice@example.com")
    monkeypatch.setenv("TEST_BROWSER_AUTH_PASSWORD", "super-secret")

    async def _no_delay(_min_ms: int = 0, _max_ms: int = 0) -> None:
        return None

    monkeypatch.setattr(auth_handlers, "_random_delay", _no_delay)

    username_loc = _FakeLocator(visible=True)
    password_loc = _FakeLocator(visible=True)
    submit_loc = _FakeLocator(visible=True)
    page = _FakePage(
        {
            "#username": username_loc,
            "#password": password_loc,
            "#submit": submit_loc,
        }
    )

    await auth_handlers._fill_form(page, _make_auth_config())

    assert username_loc.typed_text == "alice@example.com"
    assert password_loc.typed_text == "super-secret"
    assert submit_loc.click_count == 1


@pytest.mark.asyncio
async def test_fill_form_two_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_BROWSER_AUTH_USERNAME", "alice@example.com")
    monkeypatch.setenv("TEST_BROWSER_AUTH_PASSWORD", "super-secret")

    async def _no_delay(_min_ms: int = 0, _max_ms: int = 0) -> None:
        return None

    monkeypatch.setattr(auth_handlers, "_random_delay", _no_delay)

    username_loc = _FakeLocator(visible=True)
    password_loc = _FakeLocator(visible=False)
    submit_loc = _FakeLocator(visible=True)

    def _on_submit_click() -> None:
        if submit_loc.click_count == 1:
            password_loc.visible = True

    submit_loc.on_click = _on_submit_click
    page = _FakePage(
        {
            "#username": username_loc,
            "#password": password_loc,
            "#submit": submit_loc,
        }
    )

    await auth_handlers._fill_form(page, _make_auth_config())

    assert username_loc.typed_text == "alice@example.com"
    assert password_loc.typed_text == "super-secret"
    assert submit_loc.click_count == 2
    assert password_loc.pressed_keys == ["Tab"]


@pytest.mark.asyncio
async def test_wait_for_enabled_waits_until_control_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_delay(_min_ms: int = 0, _max_ms: int = 0) -> None:
        return None

    monkeypatch.setattr(auth_handlers, "_random_delay", _no_delay)

    locator = _FakeLocator(visible=True, disabled_sequence=[True, True, False])
    await auth_handlers._wait_for_enabled(locator, timeout_ms=100)
    assert locator.disabled is False


@pytest.mark.asyncio
async def test_wait_for_enabled_raises_when_still_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_delay(_min_ms: int = 0, _max_ms: int = 0) -> None:
        return None

    monkeypatch.setattr(auth_handlers, "_random_delay", _no_delay)

    locator = _FakeLocator(visible=True, disabled=True)
    with pytest.raises(TimeoutError, match="stayed disabled"):
        await auth_handlers._wait_for_enabled(locator, timeout_ms=20)


@pytest.mark.asyncio
async def test_wait_for_success_uses_url_pattern_when_matching() -> None:
    page = _FakePage(
        {
            "#username": _FakeLocator(visible=False),
            "#password": _FakeLocator(visible=False),
        },
        url="https://example.com/dashboard",
    )
    config = AuthConfig(
        selectors=AuthSelectors(
            username_input="#username",
            password_input="#password",
            submit_button="#submit",
        ),
        success_url_pattern="**/dashboard**",
    )

    await auth_handlers._wait_for_success(page, config)

    assert page.wait_for_url_calls == [("**/dashboard**", 30_000)]
    assert page.wait_for_load_state_calls == []


@pytest.mark.asyncio
async def test_wait_for_success_fallbacks_when_url_pattern_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_delay(_min_ms: int = 0, _max_ms: int = 0) -> None:
        return None

    monkeypatch.setattr(auth_handlers, "_random_delay", _no_delay)

    page = _FakePage(
        {
            "#username": _FakeLocator(visible=False),
            "#password": _FakeLocator(visible=False),
        },
        url="https://example.com/portal/home",
        wait_for_url_exc=TimeoutError("Timeout 30000ms exceeded"),
    )
    config = AuthConfig(
        selectors=AuthSelectors(
            username_input="#username",
            password_input="#password",
            submit_button="#submit",
        ),
        success_url_pattern="**/espace-client/**",
    )

    await auth_handlers._wait_for_success(page, config)

    assert page.wait_for_url_calls == [("**/espace-client/**", 30_000)]
    assert page.wait_for_load_state_calls == ["networkidle"]


@pytest.mark.asyncio
async def test_wait_for_success_raises_when_login_fields_still_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_delay(_min_ms: int = 0, _max_ms: int = 0) -> None:
        return None

    monkeypatch.setattr(auth_handlers, "_random_delay", _no_delay)

    page = _FakePage(
        {
            "#username": _FakeLocator(visible=False),
            "#password": _FakeLocator(visible=True),
        },
        url="https://example.com/auth/login",
        wait_for_url_exc=TimeoutError("Timeout 30000ms exceeded"),
    )
    config = AuthConfig(
        selectors=AuthSelectors(
            username_input="#username",
            password_input="#password",
            submit_button="#submit",
        ),
        success_url_pattern="**/espace-client/**",
    )

    with pytest.raises(TimeoutError, match="login form is still visible"):
        await auth_handlers._wait_for_success(page, config)
