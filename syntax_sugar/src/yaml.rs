use std::collections::BTreeMap;
use std::marker::PhantomData;

use serde::de::{Error, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

pub(super) fn unique_map<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct UniqueMap<V>(PhantomData<V>);
    impl<'de, V: Deserialize<'de>> Visitor<'de> for UniqueMap<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a mapping with unique names")
        }

        fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
            let mut values = BTreeMap::new();
            while let Some((name, value)) = access.next_entry::<String, V>()? {
                if values.insert(name.clone(), value).is_some() {
                    return Err(M::Error::custom(format!("duplicate name '{name}'")));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(UniqueMap(PhantomData))
}
