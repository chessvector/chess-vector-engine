use crate::features::{FeatureError, FeatureTier};
use serde::{Deserialize, Serialize};
/// License verification system for open-core business model
/// Validates subscription tiers and enables feature access
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// License key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseKey {
    pub key: String,
    pub tier: FeatureTier,
    pub expires_at: u64, // Unix timestamp
    pub issued_at: u64,
    pub customer_id: String,
    pub features: Vec<String>, // Specific features enabled
}

/// License validation result
#[derive(Debug, Clone)]
pub enum LicenseStatus {
    Valid(FeatureTier),
    Expired(u64), // Expired at timestamp
    Invalid,
    NotFound,
}

/// License verification errors
#[derive(Debug, Clone)]
pub enum LicenseError {
    InvalidKey(String),
    Expired { key: String, expired_at: u64 },
    NetworkError(String),
    InvalidFormat(String),
    FeatureNotLicensed { feature: String, tier: FeatureTier },
}

impl std::fmt::Display for LicenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseError::InvalidKey(key) => {
                write!(f, "Invalid license key: {key}")
            }
            LicenseError::Expired { key, expired_at } => {
                write!(f, "License key '{key}' expired at {expired_at}")
            }
            LicenseError::NetworkError(msg) => {
                write!(f, "Network error during license verification: {msg}")
            }
            LicenseError::InvalidFormat(msg) => {
                write!(f, "Invalid license format: {msg}")
            }
            LicenseError::FeatureNotLicensed { feature, tier } => {
                write!(f, "Feature '{feature}' not licensed for {tier:?} tier")
            }
        }
    }
}

impl std::error::Error for LicenseError {}

/// Local license cache for offline validation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LicenseCache {
    licenses: HashMap<String, LicenseKey>,
    last_verification: u64, // Last online verification timestamp
}

/// Main license verifier
pub struct LicenseVerifier {
    cache: LicenseCache,
    #[allow(dead_code)]
    verification_url: String,
    offline_mode: bool,
    cache_ttl: Duration, // How long to trust cached licenses
}

impl LicenseVerifier {
    /// Create new license verifier
    pub fn new(verification_url: String) -> Self {
        Self {
            cache: LicenseCache {
                licenses: HashMap::new(),
                last_verification: 0,
            },
            verification_url,
            offline_mode: false,
            cache_ttl: Duration::from_secs(24 * 60 * 60), // Cache for 24 hours
        }
    }

    /// Create verifier in offline mode (uses only cached licenses)
    pub fn new_offline() -> Self {
        let mut verifier = Self {
            cache: LicenseCache {
                licenses: HashMap::new(),
                last_verification: current_timestamp(),
            },
            verification_url: String::new(),
            offline_mode: true,
            cache_ttl: Duration::from_secs(30 * 24 * 60 * 60), // Longer cache for offline mode
        };

        // Pre-populate with demo licenses for testing
        verifier.add_demo_licenses();
        verifier
    }

    /// Add demo licenses for testing purposes
    fn add_demo_licenses(&mut self) {
        let demo_licenses = vec![
            LicenseKey {
                key: "DEMO-123456".to_string(),
                tier: FeatureTier::OpenSource,
                expires_at: current_timestamp() + 86400 * 30, // 30 days
                issued_at: current_timestamp(),
                customer_id: "demo-user".to_string(),
                features: vec![
                    "basic_position_encoding".to_string(),
                    "similarity_search".to_string(),
                    "opening_book".to_string(),
                ],
            },
            LicenseKey {
                key: "PREMIUM-789012".to_string(),
                tier: FeatureTier::Premium,
                expires_at: current_timestamp() + 86400 * 365, // 1 year
                issued_at: current_timestamp(),
                customer_id: "premium-user".to_string(),
                features: vec![
                    "gpu_acceleration".to_string(),
                    "ultra_fast_loading".to_string(),
                    "memory_mapped_files".to_string(),
                    "advanced_tactical_search".to_string(),
                    "pondering".to_string(),
                    "multi_pv_analysis".to_string(),
                ],
            },
            LicenseKey {
                key: "ENTERPRISE-345678".to_string(),
                tier: FeatureTier::Enterprise,
                expires_at: current_timestamp() + 86400 * 365 * 2, // 2 years
                issued_at: current_timestamp(),
                customer_id: "enterprise-user".to_string(),
                features: vec![
                    "distributed_training".to_string(),
                    "cloud_deployment".to_string(),
                    "enterprise_analytics".to_string(),
                    "custom_algorithms".to_string(),
                    "unlimited_positions".to_string(),
                ],
            },
        ];

        for license in demo_licenses {
            self.cache.licenses.insert(license.key.clone(), license);
        }
    }

    /// Load license cache from file
    pub fn load_cache<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if path.as_ref().exists() {
            let content = std::fs::read_to_string(path)?;
            self.cache = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    /// Save license cache to file
    pub fn save_cache<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(&self.cache)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Verify a license key
    pub async fn verify_license(&mut self, key: &str) -> Result<LicenseStatus, LicenseError> {
        // Check cache first
        if let Some(cached_license) = self.cache.licenses.get(key) {
            if self.is_cache_valid() && self.is_license_valid(cached_license) {
                return Ok(LicenseStatus::Valid(cached_license.tier.clone()));
            } else if !self.is_license_valid(cached_license) {
                return Ok(LicenseStatus::Expired(cached_license.expires_at));
            }
        }

        // Online verification if not in offline mode
        if !self.offline_mode {
            match self.verify_online(key).await {
                Ok(license) => {
                    // Cache the verified license
                    self.cache.licenses.insert(key.to_string(), license.clone());
                    self.cache.last_verification = current_timestamp();

                    if self.is_license_valid(&license) {
                        Ok(LicenseStatus::Valid(license.tier))
                    } else {
                        Ok(LicenseStatus::Expired(license.expires_at))
                    }
                }
                Err(e) => {
                    // Fall back to cached license if online verification fails
                    if let Some(cached_license) = self.cache.licenses.get(key) {
                        if self.is_license_valid(cached_license) {
                            Ok(LicenseStatus::Valid(cached_license.tier.clone()))
                        } else {
                            Ok(LicenseStatus::Expired(cached_license.expires_at))
                        }
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            // Offline mode - only use cache
            if let Some(cached_license) = self.cache.licenses.get(key) {
                if self.is_license_valid(cached_license) {
                    Ok(LicenseStatus::Valid(cached_license.tier.clone()))
                } else {
                    Ok(LicenseStatus::Expired(cached_license.expires_at))
                }
            } else {
                Ok(LicenseStatus::NotFound)
            }
        }
    }

    /// Add a license to the cache (for testing or manual activation)
    pub fn add_license(&mut self, license: LicenseKey) {
        self.cache.licenses.insert(license.key.clone(), license);
        self.cache.last_verification = current_timestamp();
    }

    /// Check if feature is licensed for given key
    pub async fn check_feature_license(
        &mut self,
        key: &str,
        feature: &str,
    ) -> Result<(), LicenseError> {
        let license_status = self.verify_license(key).await?;

        match license_status {
            LicenseStatus::Valid(tier) => {
                let cached_license = self
                    .cache
                    .licenses
                    .get(key)
                    .ok_or_else(|| LicenseError::InvalidKey(key.to_string()))?;

                // Check if feature is specifically enabled or tier allows it
                if cached_license.features.contains(&feature.to_string()) {
                    Ok(())
                } else {
                    // Use feature registry to check tier access
                    let registry = crate::features::FeatureRegistry::new();
                    if registry.is_feature_available(feature, &tier) {
                        Ok(())
                    } else {
                        Err(LicenseError::FeatureNotLicensed {
                            feature: feature.to_string(),
                            tier,
                        })
                    }
                }
            }
            LicenseStatus::Expired(expired_at) => Err(LicenseError::Expired {
                key: key.to_string(),
                expired_at,
            }),
            LicenseStatus::Invalid => Err(LicenseError::InvalidKey(key.to_string())),
            LicenseStatus::NotFound => Err(LicenseError::InvalidKey(key.to_string())),
        }
    }

    /// Online license verification (placeholder for actual API call)
    async fn verify_online(&self, key: &str) -> Result<LicenseKey, LicenseError> {
        // TODO: Implement actual HTTP request to license server
        // For now, return a mock response based on key format

        if key.starts_with("DEMO-") {
            Ok(LicenseKey {
                key: key.to_string(),
                tier: FeatureTier::OpenSource,
                expires_at: current_timestamp() + 86400 * 30, // 30 days
                issued_at: current_timestamp(),
                customer_id: "demo-user".to_string(),
                features: vec![
                    "basic_position_encoding".to_string(),
                    "similarity_search".to_string(),
                    "opening_book".to_string(),
                ],
            })
        } else if key.starts_with("PREMIUM-") {
            Ok(LicenseKey {
                key: key.to_string(),
                tier: FeatureTier::Premium,
                expires_at: current_timestamp() + 86400 * 365, // 1 year
                issued_at: current_timestamp(),
                customer_id: "premium-user".to_string(),
                features: vec![
                    "gpu_acceleration".to_string(),
                    "ultra_fast_loading".to_string(),
                    "memory_mapped_files".to_string(),
                    "advanced_tactical_search".to_string(),
                    "pondering".to_string(),
                    "multi_pv_analysis".to_string(),
                ],
            })
        } else if key.starts_with("ENTERPRISE-") {
            Ok(LicenseKey {
                key: key.to_string(),
                tier: FeatureTier::Enterprise,
                expires_at: current_timestamp() + 86400 * 365 * 2, // 2 years
                issued_at: current_timestamp(),
                customer_id: "enterprise-user".to_string(),
                features: vec![
                    "distributed_training".to_string(),
                    "cloud_deployment".to_string(),
                    "enterprise_analytics".to_string(),
                    "custom_algorithms".to_string(),
                    "unlimited_positions".to_string(),
                ],
            })
        } else {
            Err(LicenseError::InvalidKey(key.to_string()))
        }
    }

    /// Check if cached license data is still valid (not expired from cache perspective)
    fn is_cache_valid(&self) -> bool {
        let now = current_timestamp();
        (now - self.cache.last_verification) < self.cache_ttl.as_secs()
    }

    /// Check if license itself is valid (not expired)
    fn is_license_valid(&self, license: &LicenseKey) -> bool {
        current_timestamp() < license.expires_at
    }
}

/// Enhanced feature checker with license verification
pub struct LicensedFeatureChecker {
    verifier: LicenseVerifier,
    current_license_key: Option<String>,
    current_tier: FeatureTier,
}

impl LicensedFeatureChecker {
    /// Create new licensed feature checker
    pub fn new(verification_url: String) -> Self {
        Self {
            verifier: LicenseVerifier::new(verification_url),
            current_license_key: None,
            current_tier: FeatureTier::OpenSource,
        }
    }

    /// Create offline-only checker
    pub fn new_offline() -> Self {
        Self {
            verifier: LicenseVerifier::new_offline(),
            current_license_key: None,
            current_tier: FeatureTier::OpenSource,
        }
    }

    /// Activate license key
    pub async fn activate_license(&mut self, key: &str) -> Result<FeatureTier, LicenseError> {
        let status = self.verifier.verify_license(key).await?;

        match status {
            LicenseStatus::Valid(tier) => {
                self.current_license_key = Some(key.to_string());
                self.current_tier = tier.clone();
                Ok(tier)
            }
            LicenseStatus::Expired(expired_at) => Err(LicenseError::Expired {
                key: key.to_string(),
                expired_at,
            }),
            LicenseStatus::Invalid | LicenseStatus::NotFound => {
                Err(LicenseError::InvalidKey(key.to_string()))
            }
        }
    }

    /// Check if feature is available with current license
    pub async fn check_feature(&mut self, feature: &str) -> Result<(), FeatureError> {
        if let Some(key) = &self.current_license_key {
            match self.verifier.check_feature_license(key, feature).await {
                Ok(()) => Ok(()),
                Err(LicenseError::FeatureNotLicensed { feature, tier }) => {
                    Err(FeatureError::InsufficientTier {
                        feature,
                        required: tier,
                        current: self.current_tier.clone(),
                    })
                }
                Err(e) => Err(FeatureError::UnknownFeature(e.to_string())),
            }
        } else {
            // No license key - use basic feature registry
            let registry = crate::features::FeatureRegistry::new();
            if registry.is_feature_available(feature, &self.current_tier) {
                Ok(())
            } else if let Some(required_tier) = registry.get_feature_tier(feature) {
                Err(FeatureError::InsufficientTier {
                    feature: feature.to_string(),
                    required: required_tier.clone(),
                    current: self.current_tier.clone(),
                })
            } else {
                Err(FeatureError::UnknownFeature(feature.to_string()))
            }
        }
    }

    /// Get current tier
    pub fn get_current_tier(&self) -> &FeatureTier {
        &self.current_tier
    }

    /// Load license cache
    pub fn load_cache<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.verifier.load_cache(path)
    }

    /// Save license cache
    pub fn save_cache<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.verifier.save_cache(path)
    }
}

/// Get current Unix timestamp
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Duration extensions for convenience
#[allow(dead_code)]
trait DurationExt {
    fn from_hours(hours: u64) -> Duration;
    fn from_days(days: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn from_days(days: u64) -> Duration {
        Duration::from_secs(days * 86400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_demo_license() {
        let mut verifier = LicenseVerifier::new("https://api.example.com/license".to_string());

        let status = verifier.verify_license("DEMO-123456").await.unwrap();
        match status {
            LicenseStatus::Valid(tier) => {
                assert_eq!(tier, FeatureTier::OpenSource);
            }
            _ => panic!("Expected valid demo license"),
        }
    }

    #[tokio::test]
    async fn test_premium_license() {
        let mut verifier = LicenseVerifier::new("https://api.example.com/license".to_string());

        let status = verifier.verify_license("PREMIUM-789012").await.unwrap();
        match status {
            LicenseStatus::Valid(tier) => {
                assert_eq!(tier, FeatureTier::Premium);
            }
            _ => panic!("Expected valid premium license"),
        }
    }

    #[tokio::test]
    async fn test_enterprise_license() {
        let mut verifier = LicenseVerifier::new("https://api.example.com/license".to_string());

        let status = verifier.verify_license("ENTERPRISE-345678").await.unwrap();
        match status {
            LicenseStatus::Valid(tier) => {
                assert_eq!(tier, FeatureTier::Enterprise);
            }
            _ => panic!("Expected valid enterprise license"),
        }
    }

    #[tokio::test]
    async fn test_invalid_license() {
        let mut verifier = LicenseVerifier::new("https://api.example.com/license".to_string());

        let result = verifier.verify_license("INVALID-123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_licensed_feature_checker() {
        let mut checker = LicensedFeatureChecker::new_offline();

        // Should start as open source
        assert_eq!(checker.get_current_tier(), &FeatureTier::OpenSource);

        // Activate premium license
        let tier = checker.activate_license("PREMIUM-789012").await.unwrap();
        assert_eq!(tier, FeatureTier::Premium);

        // Should now allow premium features
        assert!(checker.check_feature("gpu_acceleration").await.is_ok());

        // Should still deny enterprise features
        assert!(checker.check_feature("distributed_training").await.is_err());
    }

    #[test]
    fn test_license_cache() {
        let mut verifier = LicenseVerifier::new_offline();

        let license = LicenseKey {
            key: "TEST-123".to_string(),
            tier: FeatureTier::Premium,
            expires_at: current_timestamp() + 86400, // 1 day
            issued_at: current_timestamp(),
            customer_id: "test-user".to_string(),
            features: vec!["gpu_acceleration".to_string()],
        };

        verifier.add_license(license);

        // Save and load cache
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        verifier.save_cache(temp_file.path()).unwrap();

        let mut new_verifier = LicenseVerifier::new_offline();
        new_verifier.load_cache(temp_file.path()).unwrap();

        // Should have the cached license
        assert!(new_verifier.cache.licenses.contains_key("TEST-123"));
    }
}
