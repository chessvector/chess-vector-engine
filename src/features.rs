/// Feature management for open-core business model
/// Controls access to commercial vs open source features
use std::collections::HashMap;

/// Available feature tiers
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FeatureTier {
    OpenSource,
    Premium,
    Enterprise,
}

/// Feature definitions and their required tiers
#[derive(Debug, Clone)]
pub struct FeatureRegistry {
    features: HashMap<String, FeatureTier>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut features = HashMap::new();

        // Open Source Features (Always Available)
        features.insert(
            "basic_position_encoding".to_string(),
            FeatureTier::OpenSource,
        );
        features.insert("similarity_search".to_string(), FeatureTier::OpenSource);
        features.insert("basic_tactical_search".to_string(), FeatureTier::OpenSource);
        features.insert("uci_basic".to_string(), FeatureTier::OpenSource);
        features.insert("opening_book".to_string(), FeatureTier::OpenSource);
        features.insert("json_training_data".to_string(), FeatureTier::OpenSource);
        features.insert("basic_persistence".to_string(), FeatureTier::OpenSource);

        // Premium Features
        features.insert("advanced_nnue".to_string(), FeatureTier::Premium);
        features.insert("gpu_acceleration".to_string(), FeatureTier::Premium);
        features.insert("ultra_fast_loading".to_string(), FeatureTier::Premium);
        features.insert("memory_mapped_files".to_string(), FeatureTier::Premium);
        features.insert("advanced_tactical_search".to_string(), FeatureTier::Premium);
        features.insert("pondering".to_string(), FeatureTier::Premium);
        features.insert("multi_pv_analysis".to_string(), FeatureTier::Premium);
        features.insert("advanced_pruning".to_string(), FeatureTier::Premium);
        features.insert("parallel_search".to_string(), FeatureTier::Premium);

        // Enterprise Features
        features.insert("distributed_training".to_string(), FeatureTier::Enterprise);
        features.insert("cloud_deployment".to_string(), FeatureTier::Enterprise);
        features.insert("enterprise_analytics".to_string(), FeatureTier::Enterprise);
        features.insert("custom_algorithms".to_string(), FeatureTier::Enterprise);
        features.insert("dedicated_support".to_string(), FeatureTier::Enterprise);
        features.insert("unlimited_positions".to_string(), FeatureTier::Enterprise);

        Self { features }
    }

    pub fn get_feature_tier(&self, feature: &str) -> Option<&FeatureTier> {
        self.features.get(feature)
    }

    pub fn is_feature_available(&self, feature: &str, current_tier: &FeatureTier) -> bool {
        match self.get_feature_tier(feature) {
            Some(required_tier) => Self::tier_includes(current_tier, required_tier),
            None => false, // Unknown features are not available
        }
    }

    /// Check if current tier includes required tier
    fn tier_includes(current: &FeatureTier, required: &FeatureTier) -> bool {
        match (current, required) {
            (FeatureTier::OpenSource, FeatureTier::OpenSource) => true,
            (FeatureTier::Premium, FeatureTier::OpenSource) => true,
            (FeatureTier::Premium, FeatureTier::Premium) => true,
            (FeatureTier::Enterprise, _) => true,
            _ => false,
        }
    }

    pub fn get_features_for_tier(&self, tier: &FeatureTier) -> Vec<String> {
        self.features
            .iter()
            .filter(|(_, required_tier)| Self::tier_includes(tier, required_tier))
            .map(|(feature, _)| feature.clone())
            .collect()
    }
}

/// Runtime feature checker
#[derive(Debug, Clone)]
pub struct FeatureChecker {
    registry: FeatureRegistry,
    current_tier: FeatureTier,
}

impl FeatureChecker {
    pub fn new(tier: FeatureTier) -> Self {
        Self {
            registry: FeatureRegistry::new(),
            current_tier: tier,
        }
    }

    pub fn check_feature(&self, feature: &str) -> Result<(), FeatureError> {
        if self
            .registry
            .is_feature_available(feature, &self.current_tier)
        {
            Ok(())
        } else {
            match self.registry.get_feature_tier(feature) {
                Some(required_tier) => Err(FeatureError::InsufficientTier {
                    feature: feature.to_string(),
                    required: required_tier.clone(),
                    current: self.current_tier.clone(),
                }),
                None => Err(FeatureError::UnknownFeature(feature.to_string())),
            }
        }
    }

    pub fn require_feature(&self, feature: &str) -> Result<(), FeatureError> {
        self.check_feature(feature)
    }

    pub fn get_current_tier(&self) -> &FeatureTier {
        &self.current_tier
    }

    pub fn upgrade_tier(&mut self, new_tier: FeatureTier) {
        self.current_tier = new_tier;
    }
}

/// Feature access errors
#[derive(Debug, Clone)]
pub enum FeatureError {
    InsufficientTier {
        feature: String,
        required: FeatureTier,
        current: FeatureTier,
    },
    UnknownFeature(String),
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureError::InsufficientTier {
                feature,
                required,
                current,
            } => {
                write!(
                    f,
                    "Feature '{}' requires {:?} tier, but current tier is {:?}. Please upgrade your subscription.",
                    feature, required, current
                )
            }
            FeatureError::UnknownFeature(feature) => {
                write!(f, "Unknown feature: '{}'", feature)
            }
        }
    }
}

impl std::error::Error for FeatureError {}

/// Macro for easy feature checking
#[macro_export]
macro_rules! require_feature {
    ($checker:expr, $feature:expr) => {
        $checker.require_feature($feature)?
    };
}

/// Conditional compilation macros for features
#[macro_export]
macro_rules! if_feature {
    ($checker:expr, $feature:expr, $code:block) => {
        if $checker.check_feature($feature).is_ok() {
            $code
        }
    };
}

#[macro_export]
macro_rules! if_feature_else {
    ($checker:expr, $feature:expr, $if_code:block, $else_code:block) => {
        if $checker.check_feature($feature).is_ok() {
            $if_code
        } else {
            $else_code
        }
    };
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FeatureChecker {
    fn default() -> Self {
        Self::new(FeatureTier::OpenSource)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_registry() {
        let registry = FeatureRegistry::new();

        // Test open source feature
        assert_eq!(
            registry.get_feature_tier("basic_position_encoding"),
            Some(&FeatureTier::OpenSource)
        );

        // Test premium feature
        assert_eq!(
            registry.get_feature_tier("gpu_acceleration"),
            Some(&FeatureTier::Premium)
        );

        // Test enterprise feature
        assert_eq!(
            registry.get_feature_tier("distributed_training"),
            Some(&FeatureTier::Enterprise)
        );
    }

    #[test]
    fn test_tier_access() {
        let registry = FeatureRegistry::new();

        // Open source tier can only access open source features
        assert!(registry.is_feature_available("basic_position_encoding", &FeatureTier::OpenSource));
        assert!(!registry.is_feature_available("gpu_acceleration", &FeatureTier::OpenSource));
        assert!(!registry.is_feature_available("distributed_training", &FeatureTier::OpenSource));

        // Premium tier can access open source and premium features
        assert!(registry.is_feature_available("basic_position_encoding", &FeatureTier::Premium));
        assert!(registry.is_feature_available("gpu_acceleration", &FeatureTier::Premium));
        assert!(!registry.is_feature_available("distributed_training", &FeatureTier::Premium));

        // Enterprise tier can access all features
        assert!(registry.is_feature_available("basic_position_encoding", &FeatureTier::Enterprise));
        assert!(registry.is_feature_available("gpu_acceleration", &FeatureTier::Enterprise));
        assert!(registry.is_feature_available("distributed_training", &FeatureTier::Enterprise));
    }

    #[test]
    fn test_feature_checker() {
        let mut checker = FeatureChecker::new(FeatureTier::OpenSource);

        // Should allow open source features
        assert!(checker.check_feature("basic_position_encoding").is_ok());

        // Should deny premium features
        assert!(checker.check_feature("gpu_acceleration").is_err());

        // Upgrade tier
        checker.upgrade_tier(FeatureTier::Premium);

        // Should now allow premium features
        assert!(checker.check_feature("gpu_acceleration").is_ok());
    }

    #[test]
    fn test_feature_error_messages() {
        let checker = FeatureChecker::new(FeatureTier::OpenSource);

        match checker.check_feature("gpu_acceleration") {
            Err(FeatureError::InsufficientTier {
                feature,
                required,
                current,
            }) => {
                assert_eq!(feature, "gpu_acceleration");
                assert_eq!(required, FeatureTier::Premium);
                assert_eq!(current, FeatureTier::OpenSource);
            }
            _ => panic!("Expected InsufficientTier error"),
        }

        match checker.check_feature("nonexistent_feature") {
            Err(FeatureError::UnknownFeature(feature)) => {
                assert_eq!(feature, "nonexistent_feature");
            }
            _ => panic!("Expected UnknownFeature error"),
        }
    }
}
