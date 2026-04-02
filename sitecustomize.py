from __future__ import annotations

import cv2
import difflib
import json
import math
import numpy as np
import os
import sys
import time
import types
from collections import deque


def _proof_log(message: str) -> None:
    print(message, flush=True)


_proof_log("SITECUSTOMIZE LOADED")


if getattr(sys, "frozen", False):
    _proof_log("PATCH INSTALL START")
    import lobby_automation
    import play
    import stage_manager
    import state_finder.main as state_finder_main
    import utils
    import window_controller

    _ORIGINAL_Q_COORD = (1740, 1000)
    _PLAY_CLICK_COOLDOWN = 3.0
    _PLAY_TEMPLATE_THRESHOLD = 0.68
    _BLOCKED_MESSAGE_PRINT_INTERVAL = 0.6
    _STATE_PRINT_INTERVAL = 1.0
    _POSTMATCH_PRESS_INTERVAL = 0.3
    _POSTMATCH_ACTION_REGION = (1280, 720, 640, 320)
    _BRAWLER_TEMPLATE_THRESHOLD = 0.7
    _BRAWLERS_OCR_REGION = (0, 160, 520, 920)
    _CATALOG_SCALE = 0.65
    _CATALOG_MAX_SCROLLS = 24
    _CATALOG_TOP_RESET_SWIPES = 6
    _CATALOG_PAGE_REPEAT_LIMIT = 2
    _STARTUP_MODE = os.environ.get("PYLA_STARTUP_MODE", "single").strip().lower()
    _ALL_THRESHOLD = max(
        0, int(os.environ.get("PYLA_ALL_BRAWLERS_THRESHOLD", "1000") or "1000")
    )
    _ORIGINAL_PLAY_PRINT = getattr(play, "print", print)
    _ORIGINAL_STAGE_PRINT = getattr(stage_manager, "print", print)
    _ORIGINAL_STATE_PRINT = getattr(state_finder_main, "print", print)
    _ORIGINAL_LOBBY_FIND_TEMPLATE_CENTER = lobby_automation.find_template_center
    _ORIGINAL_LOBBY_INIT = lobby_automation.LobbyAutomation.__init__
    _ORIGINAL_SELECT_BRAWLER = lobby_automation.LobbyAutomation.select_brawler
    _ORIGINAL_GET_MOVEMENT = play.Play.get_movement
    _ORIGINAL_UNSTUCK = play.Movement.unstuck_movement_if_needed
    _ORIGINAL_CHECK_HYPER = play.Play.check_if_hypercharge_ready
    _ORIGINAL_CHECK_SUPER = play.Play.check_if_super_ready
    _ORIGINAL_CHECK_GADGET = play.Play.check_if_gadget_ready
    _ORIGINAL_USE_HYPER = play.Movement.use_hypercharge
    _ORIGINAL_USE_SUPER = play.Movement.use_super
    _ORIGINAL_USE_GADGET = play.Movement.use_gadget
    _ORIGINAL_PLAY_MAIN = play.Play.main
    _ORIGINAL_STAGE_INIT = stage_manager.StageManager.__init__
    _ORIGINAL_START_GAME = stage_manager.StageManager.start_game
    _ORIGINAL_END_GAME = stage_manager.StageManager.end_game
    _ORIGINAL_PRESS_KEY = window_controller.WindowController.press_key
    _ORIGINAL_CLICK = window_controller.WindowController.click

    _BLOCKED_MESSAGES = {"default paths are blocked", "no movement possible ?"}
    _BLOCKED_WINDOW_SECONDS = 1.25
    _STUCK_DISPLACEMENT_PX = 4.0
    _RECOVERY_SUCCESS_DISPLACEMENT_PX = 6.0
    _blocked_log_times: deque[float] = deque(maxlen=8)
    _play_log_times: dict[str, float] = {}
    _stage_log_times: dict[str, float] = {}
    _state_log = {"message": None, "at": 0.0}
    _postmatch_labels = {
        "playagain": "PLAY AGAIN",
        "proceed": "PROCEED",
        "continue": "CONTINUE",
        "next": "NEXT",
        "okay": "OKAY",
        "ok": "OK",
    }

    # Restore the original generic bottom-right action target for Q.
    window_controller.key_coords_dict["Q"] = _ORIGINAL_Q_COORD
    if "H" in window_controller.key_coords_dict:
        window_controller.key_coords_dict["X"] = window_controller.key_coords_dict["H"]
    if "G" in window_controller.key_coords_dict:
        window_controller.key_coords_dict["R"] = window_controller.key_coords_dict["G"]

    def _normalize_move(movement: object) -> str:
        seen: set[str] = set()
        chars: list[str] = []
        for char in str(movement or "").lower():
            if char in "wasd" and char not in seen:
                chars.append(char)
                seen.add(char)
        return "".join(chars)

    def _prune_blocked_log_times(now: float) -> None:
        while _blocked_log_times and now - _blocked_log_times[0] > _BLOCKED_WINDOW_SECONDS:
            _blocked_log_times.popleft()

    def _blocked_log_count(now: float) -> int:
        _prune_blocked_log_times(now)
        return len(_blocked_log_times)

    def _crop_scaled_rgb(image: object, region: tuple[int, int, int, int]) -> tuple[np.ndarray, int, int]:
        rgb = np.array(image)
        height, width = rgb.shape[:2]
        orig_x, orig_y, orig_width, orig_height = region
        x = int(orig_x * width / state_finder_main.orig_screen_width)
        y = int(orig_y * height / state_finder_main.orig_screen_height)
        w = int(orig_width * width / state_finder_main.orig_screen_width)
        h = int(orig_height * height / state_finder_main.orig_screen_height)
        return rgb[y : y + h, x : x + w], x, y

    def _all_to_threshold_enabled() -> bool:
        return _STARTUP_MODE == "all_to_threshold"

    def _normalize_catalog_text(text: object) -> str:
        return "".join(char for char in str(text).lower() if char.isalnum())

    def _load_known_brawler_names() -> dict[str, str]:
        try:
            with open("./cfg/brawlers_info.json", "r", encoding="utf-8") as file:
                brawlers_info = json.load(file)
        except Exception:
            return {}
        return {
            _normalize_catalog_text(name): str(name).lower()
            for name in brawlers_info.keys()
        }

    _KNOWN_BRAWLERS = _load_known_brawler_names()

    def _match_brawler_name(raw_text: object, automator: object) -> str | None:
        normalized = _normalize_catalog_text(raw_text)
        if not normalized:
            return None

        try:
            normalized = _normalize_catalog_text(automator.resolve_ocr_typos(normalized))
        except Exception:
            pass

        if normalized in _KNOWN_BRAWLERS:
            return _KNOWN_BRAWLERS[normalized]

        matches = difflib.get_close_matches(normalized, _KNOWN_BRAWLERS.keys(), n=1, cutoff=0.82)
        if matches:
            return _KNOWN_BRAWLERS[matches[0]]
        return None

    def _catalog_results(screenshot: object) -> dict[str, object]:
        resized = screenshot.resize(
            (
                int(screenshot.width * _CATALOG_SCALE),
                int(screenshot.height * _CATALOG_SCALE),
            )
        )
        return utils.extract_text_and_positions(np.array(resized))

    def _pair_catalog_trophy(
        name_center: tuple[float, float], numeric_entries: list[tuple[int, tuple[float, float]]]
    ) -> int | None:
        best_value: int | None = None
        best_score: float | None = None
        for value, center in numeric_entries:
            delta_x = abs(center[0] - name_center[0])
            delta_y = center[1] - name_center[1]
            if delta_x > 220 or delta_y < -120 or delta_y > 180:
                continue
            score = abs(delta_y) + delta_x * 0.45
            if best_score is None or score < best_score:
                best_score = score
                best_value = value
        return best_value

    def _visible_brawler_candidates(automator: object, screenshot: object) -> list[dict[str, object]]:
        try:
            results = _catalog_results(screenshot)
        except Exception:
            return []

        numeric_entries: list[tuple[int, tuple[float, float]]] = []
        named_entries: list[dict[str, object]] = []
        for raw_text, position in results.items():
            normalized = _normalize_catalog_text(raw_text)
            center = position.get("center") if isinstance(position, dict) else None
            if not center:
                continue
            if normalized.isdigit():
                value = int(normalized)
                if 0 <= value <= 5000:
                    numeric_entries.append((value, center))
                continue

            matched = _match_brawler_name(raw_text, automator)
            if matched is None:
                continue
            named_entries.append({"name": matched, "center": center})

        seen_names: set[str] = set()
        candidates: list[dict[str, object]] = []
        for entry in sorted(named_entries, key=lambda item: (item["center"][1], item["center"][0])):
            name = str(entry["name"])
            if name in seen_names:
                continue
            trophies = _pair_catalog_trophy(entry["center"], numeric_entries)
            if trophies is None:
                continue
            seen_names.add(name)
            candidates.append(
                {
                    "name": name,
                    "trophies": trophies,
                    "center": entry["center"],
                }
            )
        return candidates

    def _find_template_center_on_screen(
        screenshot: object, template_path: str, threshold: float
    ) -> tuple[int, int] | None:
        screenshot_rgb = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2BGR)
        template = state_finder_main.load_template(
            template_path, screenshot_rgb.shape[1], screenshot_rgb.shape[0]
        )
        result = cv2.matchTemplate(screenshot_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        if max_value < threshold:
            return None

        template_height, template_width = template.shape[:2]
        return (
            max_location[0] + template_width // 2,
            max_location[1] + template_height // 2,
        )

    def _open_brawler_catalog(automator: object) -> bool:
        screenshot = automator.window_controller.screenshot()
        if state_finder_main.get_state(screenshot) == "brawler_selection":
            return True

        threshold = _BRAWLER_TEMPLATE_THRESHOLD
        while threshold >= 0.5:
            center = _find_template_center_on_screen(
                screenshot,
                "./state_finder/images_to_detect/brawler_menu_btn.png",
                threshold,
            )
            if center is not None:
                automator.window_controller.click(center[0], center[1], already_include_ratio=True)
                time.sleep(0.45)
                return state_finder_main.get_state(automator.window_controller.screenshot()) == "brawler_selection"
            threshold -= 0.1
        return False

    def _catalog_swipe(window: object, start_y: int, end_y: int, duration: float = 0.45) -> None:
        wr = window.width_ratio
        hr = window.height_ratio
        window.swipe(
            int(1700 * wr),
            int(start_y * hr),
            int(1700 * wr),
            int(end_y * hr),
            duration=duration,
        )

    def _reset_catalog_to_top(window: object) -> None:
        for _ in range(_CATALOG_TOP_RESET_SWIPES):
            _catalog_swipe(window, 650, 980, 0.35)
            time.sleep(0.15)

    def _select_visible_brawler(automator: object, candidate: dict[str, object]) -> bool:
        center_x, center_y = candidate["center"]
        automator.window_controller.click(
            int(center_x / _CATALOG_SCALE),
            int(center_y / _CATALOG_SCALE),
        )
        time.sleep(0.5)
        select_x, select_y = automator.coords_cfg["lobby"]["select_btn"]
        automator.window_controller.click(select_x, select_y, already_include_ratio=False)
        time.sleep(0.6)
        return state_finder_main.get_state(automator.window_controller.screenshot()) == "lobby"

    def _scan_catalog_for_threshold(
        automator: object, threshold: int
    ) -> dict[str, object] | None:
        if not _open_brawler_catalog(automator):
            return None

        _reset_catalog_to_top(automator.window_controller)
        seen_names: set[str] = set()
        repeated_pages = 0
        last_signature: tuple[str, ...] = ()

        for _ in range(_CATALOG_MAX_SCROLLS):
            screenshot = automator.window_controller.screenshot()
            candidates = _visible_brawler_candidates(automator, screenshot)
            page_signature = tuple(candidate["name"] for candidate in candidates)
            if page_signature == last_signature and page_signature:
                repeated_pages += 1
            else:
                repeated_pages = 0
            last_signature = page_signature

            for candidate in candidates:
                seen_names.add(str(candidate["name"]))
                if int(candidate["trophies"]) < threshold:
                    if _select_visible_brawler(automator, candidate):
                        return candidate
                    return None

            if repeated_pages >= _CATALOG_PAGE_REPEAT_LIMIT:
                break

            _catalog_swipe(automator.window_controller, 900, 650, 0.55)
            time.sleep(0.3)
        return None

    def _sync_all_threshold_selection(manager: object, candidate: dict[str, object]) -> None:
        if not manager.brawlers_pick_data:
            manager.brawlers_pick_data.append({})
        manager.brawlers_pick_data[0]["brawler"] = candidate["name"]
        manager.brawlers_pick_data[0]["type"] = "trophies"
        manager.brawlers_pick_data[0]["push_until"] = _ALL_THRESHOLD
        manager.brawlers_pick_data[0]["trophies"] = int(candidate["trophies"])
        manager.brawlers_pick_data[0]["wins"] = 0
        manager.brawlers_pick_data[0]["automatically_pick"] = False
        manager.Trophy_observer.current_trophies = int(candidate["trophies"])
        manager.Trophy_observer.current_wins = 0

    def _stop_after_all_threshold_completion(manager: object) -> None:
        _ORIGINAL_STAGE_PRINT("All brawlers reached the configured trophy threshold. Stopping cleanly.")
        manager.window_controller.keys_up(list("wasd"))
        manager.window_controller.close()
        sys.exit(0)

    def _find_play_button_center(screenshot: object) -> tuple[int, int] | None:
        crop_rgb, offset_x, offset_y = _crop_scaled_rgb(
            screenshot, tuple(state_finder_main.region_data["lobby_menu"])
        )
        if crop_rgb.size == 0:
            return None

        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        full_height, full_width = crop_bgr.shape[:2]
        template = state_finder_main.load_template(
            "./state_finder/images_to_detect/lobby_menu.png",
            np.array(screenshot).shape[1],
            np.array(screenshot).shape[0],
        )
        result = cv2.matchTemplate(crop_bgr, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < _PLAY_TEMPLATE_THRESHOLD:
            return None

        template_height, template_width = template.shape[:2]
        center_x = offset_x + max_loc[0] + template_width // 2
        center_y = offset_y + max_loc[1] + template_height // 2
        if not (0 <= center_x < np.array(screenshot).shape[1] and 0 <= center_y < np.array(screenshot).shape[0]):
            return None
        return center_x, center_y

    def _extract_postmatch_safe_label(screenshot: object) -> str | None:
        crop_rgb, _, _ = _crop_scaled_rgb(screenshot, _POSTMATCH_ACTION_REGION)
        if crop_rgb.size == 0:
            return None

        try:
            texts = utils.extract_text_and_positions(crop_rgb)
        except Exception:
            return None

        normalized_texts = [
            "".join(char for char in text.lower() if char.isalnum()) for text in texts.keys()
        ]
        joined = "".join(normalized_texts)
        for pattern, label in _postmatch_labels.items():
            if pattern in joined or any(pattern in text for text in normalized_texts):
                return label
        return None

    def _find_brawlers_button_by_ocr(screenshot: object) -> tuple[int, int] | None:
        crop_rgb, offset_x, offset_y = _crop_scaled_rgb(screenshot, _BRAWLERS_OCR_REGION)
        if crop_rgb.size == 0:
            return None

        try:
            texts = utils.extract_text_and_positions(crop_rgb)
        except Exception:
            return None

        best_center: tuple[int, int] | None = None
        best_score = 0.0
        for raw_text, position in texts.items():
            if not isinstance(position, dict) or "center" not in position:
                continue

            normalized = _normalize_catalog_text(raw_text)
            if not normalized:
                continue

            if "brawlers" in normalized:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, normalized, "brawlers").ratio()
            if score < 0.72 or score <= best_score:
                continue

            center_x, center_y = position["center"]
            best_center = (offset_x + int(center_x), offset_y + int(center_y))
            best_score = score

        return best_center

    def _patched_lobby_init(self: object, window_controller_obj: object) -> None:
        _ORIGINAL_LOBBY_INIT(self, window_controller_obj)
        self.select_brawler = types.MethodType(_patched_select_brawler_live, self)
        _proof_log("LobbyAutomation instance created")

    def _patched_select_brawler_live(self: object, brawler: object) -> object:
        _proof_log("about to call auto-pick select_brawler")
        _proof_log("entered live select_brawler patch")
        previous_find_template_center = lobby_automation.find_template_center

        def _direct_brawlers_lookup(
            main_img: object, template: object, threshold: float = 0.8
        ) -> tuple[int, int] | bool:
            _proof_log("entered real BRAWLERS lookup")
            result = _ORIGINAL_LOBBY_FIND_TEMPLATE_CENTER(main_img, template, threshold)
            if result or threshold > 0.500001:
                return result

            _proof_log("original template retries exhausted")
            _proof_log("trying direct OCR fallback for BRAWLERS")
            fallback_center = _find_brawlers_button_by_ocr(main_img)
            if fallback_center is not None:
                _proof_log("OCR BRAWLERS found, clicking")
                return fallback_center

            _proof_log("failed to detect BRAWLERS by template and direct OCR fallback")
            return result

        lobby_automation.find_template_center = _direct_brawlers_lookup
        try:
            return _ORIGINAL_SELECT_BRAWLER(self, brawler)
        finally:
            lobby_automation.find_template_center = previous_find_template_center

    def _release_movement_keys(bot: object) -> None:
        try:
            bot.window_controller.keys_up(list("wasd"))
        except Exception:
            pass

        bot.keys_hold = []
        try:
            bot.fix_movement_keys["toggled"] = False
            bot.fix_movement_keys["fixed"] = ""
            bot.fix_movement_keys["started_at"] = time.time()
        except Exception:
            pass

    def _choose_separation_move(current_move: str, alternate_move: str) -> str:
        base_move = current_move or alternate_move
        separation_move = _normalize_move(play.Movement.reverse_movement(base_move))
        if not separation_move or separation_move == alternate_move:
            separation_move = _normalize_move(play.Movement.reverse_movement(alternate_move))
        if not separation_move or separation_move == alternate_move:
            for source, target in (("d", "a"), ("a", "d"), ("w", "s"), ("s", "w")):
                if source in alternate_move:
                    separation_move = target
                    break
        return separation_move

    def _patched_play_print(*args: object, **kwargs: object) -> None:
        message = " ".join(str(arg) for arg in args).strip()
        now = time.time()
        if message in _BLOCKED_MESSAGES:
            _blocked_log_times.append(now)
            if now - _play_log_times.get(message, 0.0) < _BLOCKED_MESSAGE_PRINT_INTERVAL:
                return
            _play_log_times[message] = now
        _ORIGINAL_PLAY_PRINT(*args, **kwargs)

    def _patched_stage_print(*args: object, **kwargs: object) -> None:
        message = " ".join(str(arg) for arg in args).strip()
        interval = {
            "Game has ended, pressing Q": 0.75,
            "Pressed Q to return to lobby": 0.75,
        }.get(message)
        now = time.time()
        if interval and now - _stage_log_times.get(message, 0.0) < interval:
            return
        if interval:
            _stage_log_times[message] = now
        _ORIGINAL_STAGE_PRINT(*args, **kwargs)

    def _patched_state_print(*args: object, **kwargs: object) -> None:
        message = " ".join(str(arg) for arg in args).strip()
        if message.startswith("State: "):
            now = time.time()
            if (
                message == _state_log["message"]
                and now - float(_state_log["at"]) < _STATE_PRINT_INTERVAL
            ):
                return
            _state_log["message"] = message
            _state_log["at"] = now
        _ORIGINAL_STATE_PRINT(*args, **kwargs)

    def _count_hsv_pixels(image: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> int:
        if image.size == 0:
            return 0
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        return int(cv2.countNonZero(mask))

    def _ability_crop(frame: object, area: tuple[int, int, int, int], controller: object) -> np.ndarray:
        image = np.array(
            frame.crop(
                (
                    area[0] * controller.width_ratio,
                    area[1] * controller.height_ratio,
                    area[2] * controller.width_ratio,
                    area[3] * controller.height_ratio,
                )
            )
        )
        return image

    def _inner_crop(image: np.ndarray, margin_ratio: float = 0.2) -> np.ndarray:
        if image.size == 0:
            return image
        height, width = image.shape[:2]
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        if margin_x * 2 >= width or margin_y * 2 >= height:
            return image
        return image[margin_y : height - margin_y, margin_x : width - margin_x]

    def _log_ready_state(self: object, label: str, ready: bool) -> None:
        last_ready = getattr(self, "_ability_debug_ready", {})
        previous = last_ready.get(label)
        if previous != ready:
            _ORIGINAL_PLAY_PRINT(f"{label}_ready={ready}")
            last_ready[label] = ready
            self._ability_debug_ready = last_ready

    def _check_ability_ready(
        self: object,
        frame: object,
        label: str,
        area: tuple[int, int, int, int],
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
        minimum: float,
    ) -> bool:
        screenshot = _ability_crop(frame, area, self.window_controller)
        inner = _inner_crop(screenshot)
        outer_pixels = _count_hsv_pixels(screenshot, lower, upper)
        inner_pixels = _count_hsv_pixels(inner, lower, upper)
        ready = outer_pixels > minimum and inner_pixels > max(150, minimum * 0.35)
        _log_ready_state(self, label, ready)
        return ready

    def _patched_check_if_hypercharge_ready(self: object, frame: object) -> bool:
        return _check_ability_ready(
            self,
            frame,
            "hyper",
            tuple(play.hypercharge_crop_area),
            (137, 158, 159),
            (179, 255, 255),
            float(self.hypercharge_pixels_minimum),
        )

    def _patched_check_if_super_ready(self: object, frame: object) -> bool:
        return _check_ability_ready(
            self,
            frame,
            "super",
            tuple(play.super_crop_area),
            (19, 190, 232),
            (24, 240, 255),
            float(self.super_pixels_minimum),
        )

    def _patched_check_if_gadget_ready(self: object, frame: object) -> bool:
        return _check_ability_ready(
            self,
            frame,
            "gadget",
            tuple(play.gadget_crop_area),
            (57, 219, 165),
            (62, 255, 255),
            float(self.gadget_pixels_minimum),
        )

    def _dispatch_ability(
        self: object, label: str, ready_attr: str, key: str, reason: str, use_message: str
    ) -> None:
        ready = bool(getattr(self, ready_attr, False))
        _ORIGINAL_PLAY_PRINT(f"{label}_ready={ready}")
        if not ready:
            return
        _ORIGINAL_PLAY_PRINT(f"reason for attempted ability use: {reason}")
        _ORIGINAL_PLAY_PRINT(use_message)
        self.window_controller.press_key(key)
        _ORIGINAL_PLAY_PRINT(f"pressed={key}")

    def _patched_use_hypercharge(self: object) -> None:
        _dispatch_ability(
            self,
            "hyper",
            "is_hypercharge_ready",
            "X",
            "hypercharge ready and enemy in attack range",
            "Using hypercharge",
        )

    def _patched_use_super(self: object) -> None:
        _dispatch_ability(
            self,
            "super",
            "is_super_ready",
            "E",
            "super ready and enemy hittable",
            "Using super",
        )

    def _patched_use_gadget(self: object) -> None:
        _dispatch_ability(
            self,
            "gadget",
            "is_gadget_ready",
            "R",
            "gadget ready and enemy in attack range",
            "Using gadget",
        )

    def _patched_play_main(self: object, frame: object, brawler: object) -> object:
        if _all_to_threshold_enabled() and brawler:
            self.current_brawler = brawler
        return _ORIGINAL_PLAY_MAIN(self, frame, brawler)

    def _patched_stage_init(
        self: object, brawlers_data: object, lobby_automator: object, window: object
    ) -> None:
        _ORIGINAL_STAGE_INIT(self, brawlers_data, lobby_automator, window)
        self._all_threshold_initialized = False
        self._all_threshold_threshold = _ALL_THRESHOLD

    def _patched_start_game(self: object) -> object:
        if not _all_to_threshold_enabled():
            return _ORIGINAL_START_GAME(self)

        if not self.brawlers_pick_data:
            self.brawlers_pick_data = [
                {
                    "brawler": "",
                    "push_until": _ALL_THRESHOLD,
                    "trophies": 0,
                    "wins": 0,
                    "type": "trophies",
                    "automatically_pick": False,
                    "win_streak": 0,
                }
            ]

        current_trophies = self.Trophy_observer.current_trophies
        try:
            current_trophies = int(current_trophies if current_trophies != "" else 0)
        except Exception:
            current_trophies = 0

        needs_selection = (
            not getattr(self, "_all_threshold_initialized", False)
            or current_trophies >= int(getattr(self, "_all_threshold_threshold", _ALL_THRESHOLD))
        )

        if needs_selection:
            candidate = _scan_catalog_for_threshold(
                self.Lobby_automation, int(getattr(self, "_all_threshold_threshold", _ALL_THRESHOLD))
            )
            if candidate is None:
                _stop_after_all_threshold_completion(self)
            _sync_all_threshold_selection(self, candidate)
            self._all_threshold_initialized = True

        _ORIGINAL_STAGE_PRINT("state is lobby, starting game")
        self.window_controller.keys_up(list("wasd"))
        self.window_controller.press_key("Q")
        _ORIGINAL_STAGE_PRINT("Pressed Q to start a match")
        return None

    def _patched_get_movement(
        self: object,
        player_data: object,
        enemy_data: object,
        walls: object,
        brawler: object,
    ) -> object:
        now = time.time()
        player_pos = None
        try:
            player_pos = play.Movement.get_player_pos(player_data)
        except Exception:
            pass

        last_player_pos = getattr(self, "_wall_contact_last_player_pos", None)
        if player_pos is not None and last_player_pos is not None:
            self._wall_contact_last_displacement = math.dist(player_pos, last_player_pos)
        else:
            self._wall_contact_last_displacement = None
        self._wall_contact_last_player_pos = player_pos

        recovery = getattr(self, "_wall_contact_recovery", None)
        if recovery:
            self._wall_contact_blocked_count = _blocked_log_count(now)
            if recovery["phase"] == "separate":
                movement = recovery["separation_move"]
            else:
                movement = recovery["retry_move"]
            self._wall_contact_last_commanded_move = movement
            return movement

        blocked_before = len(_blocked_log_times)
        movement = _ORIGINAL_GET_MOVEMENT(self, player_data, enemy_data, walls, brawler)

        now = time.time()
        self._wall_contact_blocked_count = _blocked_log_count(now)
        normalized_move = _normalize_move(movement)
        self._wall_contact_last_commanded_move = normalized_move

        if len(_blocked_log_times) > blocked_before and normalized_move:
            self._wall_contact_last_alternate_move = normalized_move
            self._wall_contact_last_alternate_time = now

        return movement

    def _patched_unstuck_movement_if_needed(
        self: object, movement: object, current_time: float | None = None
    ) -> object:
        if current_time is None:
            current_time = time.time()

        normalized_move = _normalize_move(movement)
        recovery = getattr(self, "_wall_contact_recovery", None)
        last_displacement = getattr(self, "_wall_contact_last_displacement", None)

        if recovery:
            if recovery["phase"] == "separate":
                if current_time - recovery["phase_started_at"] < recovery["separation_duration"]:
                    return recovery["separation_move"]
                recovery["phase"] = "retry"
                recovery["phase_started_at"] = current_time
                _ORIGINAL_PLAY_PRINT("alternate direction retried")
                return recovery["retry_move"]

            if (
                last_displacement is not None
                and last_displacement > _RECOVERY_SUCCESS_DISPLACEMENT_PX
            ):
                _ORIGINAL_PLAY_PRINT("recovery success")
                _blocked_log_times.clear()
                self._wall_contact_recovery = None
                self.time_since_different_movement = current_time
                return _ORIGINAL_UNSTUCK(self, movement, current_time)

            if current_time >= recovery["timeout_at"]:
                _ORIGINAL_PLAY_PRINT("recovery timeout")
                _blocked_log_times.clear()
                self._wall_contact_recovery = None
                self.time_since_different_movement = current_time
                return _ORIGINAL_UNSTUCK(self, movement, current_time)

            return recovery["retry_move"]

        alternate_move = _normalize_move(
            getattr(self, "_wall_contact_last_alternate_move", "")
        )
        alternate_time = getattr(self, "_wall_contact_last_alternate_time", 0.0)
        blocked_count = getattr(self, "_wall_contact_blocked_count", 0)
        hold_duration = float(getattr(self, "fix_movement_keys", {}).get("duration", 0.35) or 0.35)
        separation_duration = min(0.18, max(0.08, hold_duration * 0.5))

        if (
            blocked_count >= 2
            and normalized_move
            and last_displacement is not None
            and last_displacement <= _STUCK_DISPLACEMENT_PX
            and alternate_move
            and current_time - alternate_time <= _BLOCKED_WINDOW_SECONDS
        ):
            separation_move = _choose_separation_move(normalized_move, alternate_move)
            if separation_move and separation_move != alternate_move:
                _ORIGINAL_PLAY_PRINT("entered wall_contact_stuck")
                _release_movement_keys(self)
                _blocked_log_times.clear()
                self.time_since_different_movement = current_time
                self._wall_contact_recovery = {
                    "phase": "separate",
                    "phase_started_at": current_time,
                    "separation_duration": separation_duration,
                    "separation_move": separation_move,
                    "retry_move": alternate_move,
                    "timeout_at": current_time + separation_duration + hold_duration,
                }
                _ORIGINAL_PLAY_PRINT("separation started")
                return separation_move

        return _ORIGINAL_UNSTUCK(self, movement, current_time)

    def _patched_press_key(
        self: object, key: str, delay: tuple[str, str] = ("touch_up", "touch_down")
    ) -> object:
        if key == "Q":
            screenshot = self.screenshot()
            if getattr(self, "_fast_postmatch_active", False):
                label = _extract_postmatch_safe_label(screenshot)
                now = time.time()
                if label and now - getattr(self, "_last_postmatch_press_at", 0.0) >= _POSTMATCH_PRESS_INTERVAL:
                    self._last_postmatch_press_at = now
                    _ORIGINAL_CLICK(
                        self,
                        _ORIGINAL_Q_COORD[0],
                        _ORIGINAL_Q_COORD[1],
                        already_include_ratio=False,
                    )
                    _ORIGINAL_STAGE_PRINT(f"exact safe button clicked: {label}")
                    return None
            else:
                play_center = _find_play_button_center(screenshot)
                if play_center is not None:
                    now = time.time()
                    if now - getattr(self, "_last_play_click_at", 0.0) < _PLAY_CLICK_COOLDOWN:
                        return None
                    self._last_play_click_at = now
                    _ORIGINAL_STAGE_PRINT("lobby confirmed")
                    _ORIGINAL_STAGE_PRINT("yellow PLAY acquired")
                    _ORIGINAL_CLICK(
                        self,
                        play_center[0],
                        play_center[1],
                        already_include_ratio=True,
                    )
                    _ORIGINAL_STAGE_PRINT("yellow PLAY clicked")
                    return None

        return _ORIGINAL_PRESS_KEY(self, key, delay)

    def _patched_end_game(self: object) -> object:
        screenshot = self.window_controller.screenshot()
        if stage_manager.get_state(screenshot) != "end":
            return _ORIGINAL_END_GAME(self)

        _ORIGINAL_STAGE_PRINT("entered fast_postmatch_advance")
        original_sleep = stage_manager.time.sleep
        previous_fast_postmatch = getattr(self.window_controller, "_fast_postmatch_active", False)
        self.window_controller._fast_postmatch_active = True
        self.time_since_last_stat_change = 0

        def _fast_sleep(seconds: float) -> None:
            if seconds >= 3:
                return original_sleep(0.45)
            return original_sleep(seconds)

        stage_manager.time.sleep = _fast_sleep
        try:
            return _ORIGINAL_END_GAME(self)
        finally:
            stage_manager.time.sleep = original_sleep
            self.window_controller._fast_postmatch_active = previous_fast_postmatch
            final_state = stage_manager.get_state(self.window_controller.screenshot())
            if final_state == "lobby":
                _ORIGINAL_STAGE_PRINT("returned to lobby")

    play.print = _patched_play_print
    stage_manager.print = _patched_stage_print
    state_finder_main.print = _patched_state_print
    lobby_automation.LobbyAutomation.__init__ = _patched_lobby_init
    lobby_automation.LobbyAutomation.select_brawler = _patched_select_brawler_live
    _proof_log("PATCHED LobbyAutomation.__init__")
    _proof_log("PATCHED LobbyAutomation.select_brawler")
    play.Play.check_if_hypercharge_ready = _patched_check_if_hypercharge_ready
    play.Play.check_if_super_ready = _patched_check_if_super_ready
    play.Play.check_if_gadget_ready = _patched_check_if_gadget_ready
    play.Play.main = _patched_play_main
    play.Play.get_movement = _patched_get_movement
    play.Movement.unstuck_movement_if_needed = _patched_unstuck_movement_if_needed
    play.Movement.use_hypercharge = _patched_use_hypercharge
    play.Movement.use_super = _patched_use_super
    play.Movement.use_gadget = _patched_use_gadget
    window_controller.WindowController.press_key = _patched_press_key
    stage_manager.StageManager.__init__ = _patched_stage_init
    stage_manager.StageManager.start_game = _patched_start_game
    stage_manager.StageManager.end_game = _patched_end_game
