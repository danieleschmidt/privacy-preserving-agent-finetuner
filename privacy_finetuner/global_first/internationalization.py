"""Advanced internationalization (i18n) and localization (l10n) system.

This module provides comprehensive international deployment capabilities including:
- Multi-language support with dynamic locale switching
- Cultural adaptation for different regions
- Time zone and date formatting management
- Currency and number formatting
- Right-to-left (RTL) language support
"""

import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class SupportedLocale(Enum):
    """Supported locales for international deployment."""
    EN_US = "en_US"  # English - United States
    EN_GB = "en_GB"  # English - United Kingdom
    EN_CA = "en_CA"  # English - Canada
    EN_AU = "en_AU"  # English - Australia
    FR_FR = "fr_FR"  # French - France
    FR_CA = "fr_CA"  # French - Canada
    DE_DE = "de_DE"  # German - Germany
    ES_ES = "es_ES"  # Spanish - Spain
    ES_MX = "es_MX"  # Spanish - Mexico
    IT_IT = "it_IT"  # Italian - Italy
    PT_BR = "pt_BR"  # Portuguese - Brazil
    JA_JP = "ja_JP"  # Japanese - Japan
    KO_KR = "ko_KR"  # Korean - South Korea
    ZH_CN = "zh_CN"  # Chinese - China
    ZH_TW = "zh_TW"  # Chinese - Taiwan
    AR_SA = "ar_SA"  # Arabic - Saudi Arabia
    HI_IN = "hi_IN"  # Hindi - India
    RU_RU = "ru_RU"  # Russian - Russia
    NL_NL = "nl_NL"  # Dutch - Netherlands
    SV_SE = "sv_SE"  # Swedish - Sweden


class TextDirection(Enum):
    """Text direction for different languages."""
    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left


class DateFormat(Enum):
    """Date format patterns for different regions."""
    US = "MM/dd/yyyy"  # 12/31/2023
    EUROPE = "dd/MM/yyyy"  # 31/12/2023
    ISO = "yyyy-MM-dd"  # 2023-12-31
    JAPANESE = "yyyy年MM月dd日"  # 2023年12月31日
    ARABIC = "dd/MM/yyyy"  # 31/12/2023


class NumberFormat(Enum):
    """Number formatting patterns."""
    US = "1,234.56"  # Comma thousands, period decimal
    EUROPE = "1.234,56"  # Period thousands, comma decimal
    SPACE = "1 234,56"  # Space thousands, comma decimal
    INDIAN = "1,23,456.78"  # Indian numbering system


@dataclass
class CultureSettings:
    """Cultural configuration for a specific locale."""
    locale: SupportedLocale
    language_code: str
    country_code: str
    display_name: str
    native_name: str
    text_direction: TextDirection
    date_format: DateFormat
    time_format: str  # 24-hour or 12-hour
    number_format: NumberFormat
    currency_code: str
    currency_symbol: str
    decimal_separator: str
    thousands_separator: str
    first_day_of_week: int  # 0=Sunday, 1=Monday
    timezone_preference: str
    calendar_system: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocaleConfiguration:
    """Configuration for locale-specific features."""
    locale: SupportedLocale
    culture_settings: CultureSettings
    translation_files: List[str]
    font_preferences: List[str]
    input_methods: List[str]
    keyboard_layouts: List[str]
    sorting_rules: str
    collation_strength: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class I18nManager:
    """Advanced internationalization and localization manager."""
    
    def __init__(
        self,
        default_locale: SupportedLocale = SupportedLocale.EN_US,
        fallback_locale: SupportedLocale = SupportedLocale.EN_US,
        translation_base_path: str = "translations",
        enable_auto_detection: bool = True
    ):
        """Initialize i18n manager.
        
        Args:
            default_locale: Default locale for the application
            fallback_locale: Fallback locale if translation not found
            translation_base_path: Base path for translation files
            enable_auto_detection: Enable automatic locale detection
        """
        self.default_locale = default_locale
        self.fallback_locale = fallback_locale
        self.translation_base_path = translation_base_path
        self.enable_auto_detection = enable_auto_detection
        
        # Current state
        self.current_locale = default_locale
        self.loaded_translations = {}
        self.culture_configurations = {}
        self.locale_configurations = {}
        
        # Callbacks
        self.locale_change_callbacks = {}
        
        # Thread-local storage for per-request locale
        self.local = threading.local()
        
        # Initialize culture settings
        self._initialize_culture_settings()
        
        # Initialize locale configurations
        self._initialize_locale_configurations()
        
        # Load default translations
        self._load_translations(self.default_locale)
        if self.fallback_locale != self.default_locale:
            self._load_translations(self.fallback_locale)
        
        logger.info(f"I18nManager initialized with default locale: {default_locale.value}")
    
    def _initialize_culture_settings(self) -> None:
        """Initialize culture settings for supported locales."""
        
        # English - United States
        self.culture_configurations[SupportedLocale.EN_US] = CultureSettings(
            locale=SupportedLocale.EN_US,
            language_code="en",
            country_code="US",
            display_name="English (United States)",
            native_name="English (United States)",
            text_direction=TextDirection.LTR,
            date_format=DateFormat.US,
            time_format="12-hour",
            number_format=NumberFormat.US,
            currency_code="USD",
            currency_symbol="$",
            decimal_separator=".",
            thousands_separator=",",
            first_day_of_week=0,  # Sunday
            timezone_preference="America/New_York",
            calendar_system="gregorian"
        )
        
        # German - Germany
        self.culture_configurations[SupportedLocale.DE_DE] = CultureSettings(
            locale=SupportedLocale.DE_DE,
            language_code="de",
            country_code="DE",
            display_name="German (Germany)",
            native_name="Deutsch (Deutschland)",
            text_direction=TextDirection.LTR,
            date_format=DateFormat.EUROPE,
            time_format="24-hour",
            number_format=NumberFormat.EUROPE,
            currency_code="EUR",
            currency_symbol="€",
            decimal_separator=",",
            thousands_separator=".",
            first_day_of_week=1,  # Monday
            timezone_preference="Europe/Berlin",
            calendar_system="gregorian"
        )
        
        # Japanese - Japan
        self.culture_configurations[SupportedLocale.JA_JP] = CultureSettings(
            locale=SupportedLocale.JA_JP,
            language_code="ja",
            country_code="JP",
            display_name="Japanese (Japan)",
            native_name="日本語 (日本)",
            text_direction=TextDirection.LTR,
            date_format=DateFormat.JAPANESE,
            time_format="24-hour",
            number_format=NumberFormat.US,  # Japan uses US-style numbers
            currency_code="JPY",
            currency_symbol="¥",
            decimal_separator=".",
            thousands_separator=",",
            first_day_of_week=0,  # Sunday
            timezone_preference="Asia/Tokyo",
            calendar_system="gregorian"
        )
        
        # Arabic - Saudi Arabia
        self.culture_configurations[SupportedLocale.AR_SA] = CultureSettings(
            locale=SupportedLocale.AR_SA,
            language_code="ar",
            country_code="SA",
            display_name="Arabic (Saudi Arabia)",
            native_name="العربية (المملكة العربية السعودية)",
            text_direction=TextDirection.RTL,
            date_format=DateFormat.ARABIC,
            time_format="12-hour",
            number_format=NumberFormat.US,
            currency_code="SAR",
            currency_symbol="ر.س",
            decimal_separator=".",
            thousands_separator=",",
            first_day_of_week=6,  # Saturday
            timezone_preference="Asia/Riyadh",
            calendar_system="islamic"
        )
        
        # French - France
        self.culture_configurations[SupportedLocale.FR_FR] = CultureSettings(
            locale=SupportedLocale.FR_FR,
            language_code="fr",
            country_code="FR",
            display_name="French (France)",
            native_name="Français (France)",
            text_direction=TextDirection.LTR,
            date_format=DateFormat.EUROPE,
            time_format="24-hour",
            number_format=NumberFormat.SPACE,
            currency_code="EUR",
            currency_symbol="€",
            decimal_separator=",",
            thousands_separator=" ",
            first_day_of_week=1,  # Monday
            timezone_preference="Europe/Paris",
            calendar_system="gregorian"
        )
        
        # Chinese - China
        self.culture_configurations[SupportedLocale.ZH_CN] = CultureSettings(
            locale=SupportedLocale.ZH_CN,
            language_code="zh",
            country_code="CN",
            display_name="Chinese (China)",
            native_name="中文 (中国)",
            text_direction=TextDirection.LTR,
            date_format=DateFormat.ISO,
            time_format="24-hour",
            number_format=NumberFormat.US,
            currency_code="CNY",
            currency_symbol="¥",
            decimal_separator=".",
            thousands_separator=",",
            first_day_of_week=1,  # Monday
            timezone_preference="Asia/Shanghai",
            calendar_system="gregorian"
        )
        
        logger.debug(f"Initialized culture settings for {len(self.culture_configurations)} locales")
    
    def _initialize_locale_configurations(self) -> None:
        """Initialize locale-specific configurations."""
        
        for locale, culture in self.culture_configurations.items():
            font_prefs = ["Arial", "Helvetica", "sans-serif"]  # Default
            input_methods = ["standard"]
            keyboard_layouts = ["qwerty"]
            
            # Locale-specific configurations
            if culture.language_code == "ja":
                font_prefs = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic"]
                input_methods = ["romaji", "hiragana", "katakana", "kanji"]
                keyboard_layouts = ["qwerty", "jis"]
            elif culture.language_code == "ar":
                font_prefs = ["Noto Sans Arabic", "Tahoma", "Arial Unicode MS"]
                input_methods = ["arabic", "arabic_transliteration"]
                keyboard_layouts = ["arabic", "qwerty"]
            elif culture.language_code == "zh":
                font_prefs = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]
                input_methods = ["pinyin", "wubi", "cangjie"]
                keyboard_layouts = ["qwerty", "pinyin"]
            elif culture.language_code == "de":
                keyboard_layouts = ["qwertz", "qwerty"]
            elif culture.language_code == "fr":
                keyboard_layouts = ["azerty", "qwerty"]
            
            self.locale_configurations[locale] = LocaleConfiguration(
                locale=locale,
                culture_settings=culture,
                translation_files=[f"{culture.language_code}.json", f"{locale.value}.json"],
                font_preferences=font_prefs,
                input_methods=input_methods,
                keyboard_layouts=keyboard_layouts,
                sorting_rules=f"sort_{culture.language_code}",
                collation_strength="tertiary"
            )
    
    def _load_translations(self, locale: SupportedLocale) -> None:
        """Load translations for specified locale."""
        if locale in self.loaded_translations:
            return
        
        culture = self.culture_configurations.get(locale)
        if not culture:
            logger.warning(f"No culture configuration for locale: {locale}")
            return
        
        translations = {}
        
        # Load translation files
        for translation_file in self.locale_configurations[locale].translation_files:
            file_path = os.path.join(self.translation_base_path, translation_file)
            
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_translations = json.load(f)
                        translations.update(file_translations)
                else:
                    # Create sample translations for demo
                    translations.update(self._create_sample_translations(locale))
            except Exception as e:
                logger.warning(f"Failed to load translations from {file_path}: {e}")
                translations.update(self._create_sample_translations(locale))
        
        self.loaded_translations[locale] = translations
        logger.info(f"Loaded {len(translations)} translations for {locale.value}")
    
    def _create_sample_translations(self, locale: SupportedLocale) -> Dict[str, str]:
        """Create sample translations for demonstration."""
        culture = self.culture_configurations[locale]
        
        base_translations = {
            "app.title": "Privacy-Preserving ML Framework",
            "app.welcome": "Welcome",
            "app.goodbye": "Goodbye",
            "privacy.title": "Privacy Settings",
            "privacy.consent": "I consent to data processing",
            "privacy.policy": "Privacy Policy",
            "data.processing": "Data Processing",
            "data.retention": "Data Retention",
            "compliance.gdpr": "GDPR Compliance",
            "compliance.status": "Compliance Status",
            "error.general": "An error occurred",
            "error.permission": "Permission denied",
            "success.saved": "Successfully saved",
            "button.save": "Save",
            "button.cancel": "Cancel",
            "button.delete": "Delete",
            "nav.home": "Home",
            "nav.settings": "Settings",
            "nav.help": "Help"
        }
        
        # Language-specific translations (simplified examples)
        if culture.language_code == "de":
            return {
                "app.title": "Datenschutzorientiertes ML-Framework",
                "app.welcome": "Willkommen",
                "app.goodbye": "Auf Wiedersehen",
                "privacy.title": "Datenschutz-Einstellungen",
                "privacy.consent": "Ich stimme der Datenverarbeitung zu",
                "privacy.policy": "Datenschutzrichtlinie",
                "data.processing": "Datenverarbeitung",
                "data.retention": "Datenspeicherung",
                "compliance.gdpr": "DSGVO-Konformität",
                "compliance.status": "Compliance-Status",
                "error.general": "Ein Fehler ist aufgetreten",
                "error.permission": "Berechtigung verweigert",
                "success.saved": "Erfolgreich gespeichert",
                "button.save": "Speichern",
                "button.cancel": "Abbrechen",
                "button.delete": "Löschen",
                "nav.home": "Startseite",
                "nav.settings": "Einstellungen",
                "nav.help": "Hilfe"
            }
        elif culture.language_code == "fr":
            return {
                "app.title": "Framework ML Préservant la Confidentialité",
                "app.welcome": "Bienvenue",
                "app.goodbye": "Au revoir",
                "privacy.title": "Paramètres de Confidentialité",
                "privacy.consent": "Je consens au traitement des données",
                "privacy.policy": "Politique de Confidentialité",
                "data.processing": "Traitement des Données",
                "data.retention": "Rétention des Données",
                "compliance.gdpr": "Conformité RGPD",
                "compliance.status": "État de Conformité",
                "error.general": "Une erreur s'est produite",
                "error.permission": "Permission refusée",
                "success.saved": "Enregistré avec succès",
                "button.save": "Enregistrer",
                "button.cancel": "Annuler",
                "button.delete": "Supprimer",
                "nav.home": "Accueil",
                "nav.settings": "Paramètres",
                "nav.help": "Aide"
            }
        elif culture.language_code == "ja":
            return {
                "app.title": "プライバシー保護ML フレームワーク",
                "app.welcome": "ようこそ",
                "app.goodbye": "さようなら",
                "privacy.title": "プライバシー設定",
                "privacy.consent": "データ処理に同意します",
                "privacy.policy": "プライバシーポリシー",
                "data.processing": "データ処理",
                "data.retention": "データ保持",
                "compliance.gdpr": "GDPR コンプライアンス",
                "compliance.status": "コンプライアンス状態",
                "error.general": "エラーが発生しました",
                "error.permission": "権限が拒否されました",
                "success.saved": "正常に保存されました",
                "button.save": "保存",
                "button.cancel": "キャンセル",
                "button.delete": "削除",
                "nav.home": "ホーム",
                "nav.settings": "設定",
                "nav.help": "ヘルプ"
            }
        elif culture.language_code == "ar":
            return {
                "app.title": "إطار عمل التعلم الآلي المحافظ على الخصوصية",
                "app.welcome": "مرحبا",
                "app.goodbye": "وداعا",
                "privacy.title": "إعدادات الخصوصية",
                "privacy.consent": "أوافق على معالجة البيانات",
                "privacy.policy": "سياسة الخصوصية",
                "data.processing": "معالجة البيانات",
                "data.retention": "الاحتفاظ بالبيانات",
                "compliance.gdpr": "امتثال اللائحة العامة لحماية البيانات",
                "compliance.status": "حالة الامتثال",
                "error.general": "حدث خطأ",
                "error.permission": "تم رفض الإذن",
                "success.saved": "تم الحفظ بنجاح",
                "button.save": "حفظ",
                "button.cancel": "إلغاء",
                "button.delete": "حذف",
                "nav.home": "الرئيسية",
                "nav.settings": "الإعدادات",
                "nav.help": "مساعدة"
            }
        elif culture.language_code == "zh":
            return {
                "app.title": "隐私保护机器学习框架",
                "app.welcome": "欢迎",
                "app.goodbye": "再见",
                "privacy.title": "隐私设置",
                "privacy.consent": "我同意数据处理",
                "privacy.policy": "隐私政策",
                "data.processing": "数据处理",
                "data.retention": "数据保留",
                "compliance.gdpr": "GDPR 合规",
                "compliance.status": "合规状态",
                "error.general": "发生错误",
                "error.permission": "权限被拒绝",
                "success.saved": "保存成功",
                "button.save": "保存",
                "button.cancel": "取消",
                "button.delete": "删除",
                "nav.home": "主页",
                "nav.settings": "设置",
                "nav.help": "帮助"
            }
        else:
            return base_translations
    
    def set_locale(self, locale: SupportedLocale) -> bool:
        """Set the current locale."""
        if locale not in self.culture_configurations:
            logger.warning(f"Unsupported locale: {locale}")
            return False
        
        previous_locale = self.current_locale
        self.current_locale = locale
        
        # Load translations if not already loaded
        if locale not in self.loaded_translations:
            self._load_translations(locale)
        
        # Trigger locale change callbacks
        for callback in self.locale_change_callbacks.values():
            try:
                callback(previous_locale, locale)
            except Exception as e:
                logger.error(f"Locale change callback failed: {e}")
        
        logger.info(f"Locale changed from {previous_locale.value} to {locale.value}")
        return True
    
    def get_current_locale(self) -> SupportedLocale:
        """Get the current locale."""
        # Check thread-local storage first
        if hasattr(self.local, 'locale'):
            return self.local.locale
        return self.current_locale
    
    def set_thread_locale(self, locale: SupportedLocale) -> bool:
        """Set locale for current thread."""
        if locale not in self.culture_configurations:
            return False
        
        if locale not in self.loaded_translations:
            self._load_translations(locale)
        
        self.local.locale = locale
        return True
    
    def translate(self, key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
        """Translate a key to the specified or current locale."""
        target_locale = locale or self.get_current_locale()
        
        # Try target locale
        translations = self.loaded_translations.get(target_locale, {})
        if key in translations:
            message = translations[key]
        else:
            # Try fallback locale
            fallback_translations = self.loaded_translations.get(self.fallback_locale, {})
            if key in fallback_translations:
                message = fallback_translations[key]
                logger.debug(f"Used fallback translation for key: {key}")
            else:
                # Return key as last resort
                message = key
                logger.warning(f"No translation found for key: {key}")
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                message = message.format(**kwargs)
            except Exception as e:
                logger.warning(f"Translation formatting failed for key {key}: {e}")
        
        return message
    
    def get_culture_settings(self, locale: Optional[SupportedLocale] = None) -> CultureSettings:
        """Get culture settings for specified or current locale."""
        target_locale = locale or self.get_current_locale()
        return self.culture_configurations.get(target_locale, self.culture_configurations[self.fallback_locale])
    
    def format_date(self, timestamp: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format date according to locale preferences."""
        target_locale = locale or self.get_current_locale()
        culture = self.get_culture_settings(target_locale)
        
        import time
        date_struct = time.localtime(timestamp)
        
        if culture.date_format == DateFormat.US:
            return f"{date_struct.tm_mon:02d}/{date_struct.tm_mday:02d}/{date_struct.tm_year}"
        elif culture.date_format == DateFormat.EUROPE:
            return f"{date_struct.tm_mday:02d}/{date_struct.tm_mon:02d}/{date_struct.tm_year}"
        elif culture.date_format == DateFormat.ISO:
            return f"{date_struct.tm_year}-{date_struct.tm_mon:02d}-{date_struct.tm_mday:02d}"
        elif culture.date_format == DateFormat.JAPANESE:
            return f"{date_struct.tm_year}年{date_struct.tm_mon}月{date_struct.tm_mday}日"
        else:
            return f"{date_struct.tm_mday:02d}/{date_struct.tm_mon:02d}/{date_struct.tm_year}"
    
    def format_time(self, timestamp: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format time according to locale preferences."""
        target_locale = locale or self.get_current_locale()
        culture = self.get_culture_settings(target_locale)
        
        import time
        time_struct = time.localtime(timestamp)
        
        if culture.time_format == "12-hour":
            hour = time_struct.tm_hour
            period = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            display_hour = 12 if display_hour == 0 else display_hour
            return f"{display_hour}:{time_struct.tm_min:02d} {period}"
        else:  # 24-hour
            return f"{time_struct.tm_hour:02d}:{time_struct.tm_min:02d}"
    
    def format_number(self, number: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format number according to locale preferences."""
        target_locale = locale or self.get_current_locale()
        culture = self.get_culture_settings(target_locale)
        
        # Split into integer and decimal parts
        integer_part = int(abs(number))
        decimal_part = abs(number) - integer_part
        
        # Format integer part with thousands separator
        integer_str = str(integer_part)
        
        if culture.number_format == NumberFormat.US:
            # Add commas every 3 digits from right
            if len(integer_str) > 3:
                parts = []
                for i, digit in enumerate(reversed(integer_str)):
                    if i > 0 and i % 3 == 0:
                        parts.append(',')
                    parts.append(digit)
                integer_str = ''.join(reversed(parts))
        elif culture.number_format == NumberFormat.EUROPE:
            # Add periods every 3 digits from right
            if len(integer_str) > 3:
                parts = []
                for i, digit in enumerate(reversed(integer_str)):
                    if i > 0 and i % 3 == 0:
                        parts.append('.')
                    parts.append(digit)
                integer_str = ''.join(reversed(parts))
        elif culture.number_format == NumberFormat.SPACE:
            # Add spaces every 3 digits from right
            if len(integer_str) > 3:
                parts = []
                for i, digit in enumerate(reversed(integer_str)):
                    if i > 0 and i % 3 == 0:
                        parts.append(' ')
                    parts.append(digit)
                integer_str = ''.join(reversed(parts))
        
        # Format decimal part
        decimal_str = ""
        if decimal_part > 0:
            decimal_digits = f"{decimal_part:.2f}"[2:]  # Get digits after decimal
            decimal_str = culture.decimal_separator + decimal_digits
        
        # Add sign
        sign = "-" if number < 0 else ""
        
        return sign + integer_str + decimal_str
    
    def format_currency(self, amount: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format currency according to locale preferences."""
        target_locale = locale or self.get_current_locale()
        culture = self.get_culture_settings(target_locale)
        
        formatted_number = self.format_number(amount, target_locale)
        
        # Currency symbol placement varies by locale
        if culture.language_code in ["en", "ja", "zh"]:
            return f"{culture.currency_symbol}{formatted_number}"
        elif culture.language_code in ["de", "fr"]:
            return f"{formatted_number} {culture.currency_symbol}"
        elif culture.language_code == "ar":
            return f"{formatted_number} {culture.currency_symbol}"
        else:
            return f"{culture.currency_symbol}{formatted_number}"
    
    def auto_detect_locale(self, headers: Dict[str, str]) -> Optional[SupportedLocale]:
        """Auto-detect locale from HTTP headers or environment."""
        if not self.enable_auto_detection:
            return None
        
        # Check Accept-Language header
        accept_language = headers.get('Accept-Language', '')
        if accept_language:
            # Parse Accept-Language header (simplified)
            languages = accept_language.split(',')
            for lang_entry in languages:
                lang = lang_entry.split(';')[0].strip()
                
                # Try exact match first
                for locale in SupportedLocale:
                    if locale.value.replace('_', '-').lower() == lang.lower():
                        return locale
                
                # Try language match
                lang_code = lang.split('-')[0].lower()
                for locale in SupportedLocale:
                    locale_lang = locale.value.split('_')[0].lower()
                    if locale_lang == lang_code:
                        return locale
        
        return None
    
    def get_rtl_locales(self) -> List[SupportedLocale]:
        """Get list of right-to-left locales."""
        return [
            locale for locale, culture in self.culture_configurations.items()
            if culture.text_direction == TextDirection.RTL
        ]
    
    def is_rtl_locale(self, locale: Optional[SupportedLocale] = None) -> bool:
        """Check if locale uses right-to-left text direction."""
        target_locale = locale or self.get_current_locale()
        culture = self.get_culture_settings(target_locale)
        return culture.text_direction == TextDirection.RTL
    
    def register_locale_change_callback(self, name: str, callback: Callable[[SupportedLocale, SupportedLocale], None]) -> None:
        """Register callback for locale changes."""
        self.locale_change_callbacks[name] = callback
        logger.info(f"Registered locale change callback: {name}")
    
    def get_supported_locales(self) -> List[SupportedLocale]:
        """Get list of all supported locales."""
        return list(self.culture_configurations.keys())
    
    def get_locale_display_name(self, locale: SupportedLocale, display_locale: Optional[SupportedLocale] = None) -> str:
        """Get display name for locale in specified language."""
        target_display_locale = display_locale or self.get_current_locale()
        
        # For demo, return native name
        culture = self.culture_configurations.get(locale)
        if culture:
            return culture.native_name
        return locale.value
    
    def export_translations(self, locale: SupportedLocale, output_path: str) -> None:
        """Export translations for specified locale."""
        if locale not in self.loaded_translations:
            self._load_translations(locale)
        
        translations = self.loaded_translations.get(locale, {})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(translations)} translations to {output_path}")
    
    def import_translations(self, locale: SupportedLocale, file_path: str) -> bool:
        """Import translations from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            if locale not in self.loaded_translations:
                self.loaded_translations[locale] = {}
            
            self.loaded_translations[locale].update(translations)
            logger.info(f"Imported {len(translations)} translations for {locale.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import translations: {e}")
            return False
    
    def get_translation_completeness(self, locale: SupportedLocale) -> Dict[str, Any]:
        """Get translation completeness statistics."""
        if locale not in self.loaded_translations:
            return {"completeness": 0.0, "missing_keys": [], "total_keys": 0}
        
        base_translations = self.loaded_translations.get(self.fallback_locale, {})
        target_translations = self.loaded_translations.get(locale, {})
        
        base_keys = set(base_translations.keys())
        target_keys = set(target_translations.keys())
        
        missing_keys = list(base_keys - target_keys)
        completeness = len(target_keys) / len(base_keys) * 100 if base_keys else 100
        
        return {
            "completeness": completeness,
            "missing_keys": missing_keys,
            "total_keys": len(base_keys),
            "translated_keys": len(target_keys),
            "missing_count": len(missing_keys)
        }
    
    def generate_i18n_report(self) -> Dict[str, Any]:
        """Generate comprehensive internationalization report."""
        report = {
            "report_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_locale": self.current_locale.value,
            "default_locale": self.default_locale.value,
            "fallback_locale": self.fallback_locale.value,
            "supported_locales_count": len(self.culture_configurations),
            "loaded_locales_count": len(self.loaded_translations),
            "locale_details": {},
            "translation_completeness": {},
            "rtl_locales": [loc.value for loc in self.get_rtl_locales()],
            "culture_coverage": {}
        }
        
        # Locale details
        for locale, culture in self.culture_configurations.items():
            report["locale_details"][locale.value] = {
                "display_name": culture.display_name,
                "native_name": culture.native_name,
                "language_code": culture.language_code,
                "country_code": culture.country_code,
                "text_direction": culture.text_direction.value,
                "currency": f"{culture.currency_code} ({culture.currency_symbol})",
                "loaded": locale in self.loaded_translations
            }
            
            # Translation completeness
            completeness = self.get_translation_completeness(locale)
            report["translation_completeness"][locale.value] = completeness
        
        # Culture coverage
        languages = set(culture.language_code for culture in self.culture_configurations.values())
        countries = set(culture.country_code for culture in self.culture_configurations.values())
        
        report["culture_coverage"] = {
            "languages_supported": len(languages),
            "countries_supported": len(countries),
            "languages": list(languages),
            "countries": list(countries)
        }
        
        return report