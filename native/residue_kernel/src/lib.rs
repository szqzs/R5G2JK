use std::cmp;
use std::collections::HashMap;
use std::ffi::CString;
use std::hash::Hash;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

type Alpha = [i32; 4];
type DerivOrders = [i32; 4];
type Denom = [u32; 10];
type DenomKey = u128;
type ProductKey = (i32, i32, i32, i32);
type CompactTerms3Key = (i32, i32, i32, DenomKey);
type CompactTerms2Key = (i32, i32, DenomKey);
type CompactTerms1Key = (i32, DenomKey);
type Y3GroupedItem = (i32, DenomKey, i32, i32, u64);
type Y2GroupedItem = (i32, DenomKey, i32, u64);

const ZERO_DENOM: Denom = [0; 10];
const ROOT_POWERS: Denom = [2; 10];
const SIMPLE_ROOT_POS: [usize; 4] = [0, 4, 7, 9];
const BASE_LAMBDA_NUMS: [i32; 4] = [-1, -2, -3, -4];
const ROOT_TRANSITIONS: [&[(usize, i32)]; 4] = [
    &[(0, -1)],
    &[(1, 0), (4, -1)],
    &[(2, 1), (5, 4), (7, -1)],
    &[(3, 2), (6, 5), (8, 7), (9, -1)],
];

#[derive(Clone, PartialEq, Eq, Hash)]
struct VarKey {
    var_idx: u8,
    deriv_orders: DerivOrders,
    y_exp: i32,
    denom_powers: Denom,
    p: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SpecialKey {
    order: i32,
    lam_num: i32,
    cutoff: i32,
    p: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct BinomKey {
    root_power: u32,
    cutoff: i32,
    p: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct HKey {
    nmax: i32,
    p: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ExpKey {
    lam_num: i32,
    nmax: i32,
    p: u64,
}

#[derive(Default)]
struct Caches {
    variable: HashMap<VarKey, Vec<(Denom, u64)>>,
    special: HashMap<SpecialKey, Vec<(i32, u64)>>,
    binomial: HashMap<BinomKey, Vec<u64>>,
    h_coeffs: HashMap<HKey, Vec<u64>>,
    exp_coeffs: HashMap<ExpKey, Vec<u64>>,
}

#[derive(Default)]
struct KernelProfile {
    value: u64,
    tasks: u64,
    left_terms: u64,
    right_terms: u64,
    pair_products: u64,
    product_terms_total: u64,
    product_terms_max: usize,
    variable_transition_calls: u64,
    variable_cache_hits: u64,
    variable_cache_misses: u64,
    product_y4_nanos: u128,
    y3_nanos: u128,
    y2_nanos: u128,
    y1_nanos: u128,
    right_vec_build_nanos: u128,
    terms3_total: u64,
    terms2_total: u64,
    terms1_total: u64,
    terms3_max: usize,
    terms2_max: usize,
    terms1_max: usize,
    y3_groups_total: u64,
    y2_groups_total: u64,
    y3_groups_max: usize,
    y2_groups_max: usize,
}

impl KernelProfile {
    fn observe_product_terms(&mut self, len: usize) {
        self.product_terms_total += len as u64;
        self.product_terms_max = cmp::max(self.product_terms_max, len);
    }

    fn observe_terms3(&mut self, len: usize) {
        self.terms3_total += len as u64;
        self.terms3_max = cmp::max(self.terms3_max, len);
    }

    fn observe_terms2(&mut self, len: usize) {
        self.terms2_total += len as u64;
        self.terms2_max = cmp::max(self.terms2_max, len);
    }

    fn observe_terms1(&mut self, len: usize) {
        self.terms1_total += len as u64;
        self.terms1_max = cmp::max(self.terms1_max, len);
    }

    fn observe_y3_groups(&mut self, len: usize) {
        self.y3_groups_total += len as u64;
        self.y3_groups_max = cmp::max(self.y3_groups_max, len);
    }

    fn observe_y2_groups(&mut self, len: usize) {
        self.y2_groups_total += len as u64;
        self.y2_groups_max = cmp::max(self.y2_groups_max, len);
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"value\":{},",
                "\"tasks\":{},",
                "\"left_terms\":{},",
                "\"right_terms\":{},",
                "\"pair_products\":{},",
                "\"product_terms_total\":{},",
                "\"product_terms_max\":{},",
                "\"variable_transition_calls\":{},",
                "\"variable_cache_hits\":{},",
                "\"variable_cache_misses\":{},",
                "\"product_y4_nanos\":{},",
                "\"y3_nanos\":{},",
                "\"y2_nanos\":{},",
                "\"y1_nanos\":{},",
                "\"right_vec_build_nanos\":{},",
                "\"terms3_total\":{},",
                "\"terms2_total\":{},",
                "\"terms1_total\":{},",
                "\"terms3_max\":{},",
                "\"terms2_max\":{},",
                "\"terms1_max\":{},",
                "\"y3_groups_total\":{},",
                "\"y2_groups_total\":{},",
                "\"y3_groups_max\":{},",
                "\"y2_groups_max\":{}",
                "}}"
            ),
            self.value,
            self.tasks,
            self.left_terms,
            self.right_terms,
            self.pair_products,
            self.product_terms_total,
            self.product_terms_max,
            self.variable_transition_calls,
            self.variable_cache_hits,
            self.variable_cache_misses,
            self.product_y4_nanos,
            self.y3_nanos,
            self.y2_nanos,
            self.y1_nanos,
            self.right_vec_build_nanos,
            self.terms3_total,
            self.terms2_total,
            self.terms1_total,
            self.terms3_max,
            self.terms2_max,
            self.terms1_max,
            self.y3_groups_total,
            self.y2_groups_total,
            self.y3_groups_max,
            self.y2_groups_max,
        )
    }
}

static CACHES: OnceLock<Mutex<Caches>> = OnceLock::new();

fn caches() -> &'static Mutex<Caches> {
    CACHES.get_or_init(|| Mutex::new(Caches::default()))
}

fn mod_from_i128(value: i128, p: u64) -> u64 {
    let modulus = p as i128;
    let mut out = value % modulus;
    if out < 0 {
        out += modulus;
    }
    out as u64
}

fn add_assign_mod(slot: &mut u64, value: u64, p: u64) {
    *slot = ((*slot as u128 + value as u128) % p as u128) as u64;
}

fn add_to_map<K: Eq + Hash>(map: &mut HashMap<K, u64>, key: K, value: u64, p: u64) {
    if value == 0 {
        return;
    }
    let slot = map.entry(key).or_insert(0);
    add_assign_mod(slot, value, p);
}

fn pack_denom(denom: Denom) -> DenomKey {
    let mut out = 0_u128;
    for (idx, power) in denom.into_iter().enumerate() {
        debug_assert!(power <= 4095);
        out |= (power as u128) << (12 * idx);
    }
    out
}

fn unpack_denom(key: DenomKey) -> Denom {
    let mut out = [0_u32; 10];
    for (idx, slot) in out.iter_mut().enumerate() {
        *slot = ((key >> (12 * idx)) & 0xfff) as u32;
    }
    out
}

fn mul_mod(left: u64, right: u64, p: u64) -> u64 {
    ((left as u128 * right as u128) % p as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut out = 1_u64;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            out = mul_mod(out, base, p);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod(base, base, p);
        }
    }
    out
}

fn mod_inv(value: u64, p: u64) -> u64 {
    let value = value % p;
    if value == 0 {
        return 0;
    }
    mod_pow(value, p - 2, p)
}

fn factorial_mod(n: i32, p: u64) -> u64 {
    let mut out = 1_u64;
    for k in 2..=n {
        out = mul_mod(out, k as u64, p);
    }
    out
}

fn comb_mod(n: u32, k: u32, p: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = cmp::min(k, n - k);
    let mut out = 1_u64;
    for i in 1..=k {
        out = mul_mod(out, (n - k + i) as u64 % p, p);
        out = mul_mod(out, mod_inv(i as u64, p), p);
    }
    out
}

fn h_coeffs_mod(nmax: i32, p: u64, caches: &mut Caches) -> Vec<u64> {
    let key = HKey { nmax, p };
    if let Some(value) = caches.h_coeffs.get(&key) {
        return value.clone();
    }
    let mut out = Vec::with_capacity((nmax + 1).max(0) as usize);
    let mut fact = 1_u64;
    for n in 0..=nmax {
        fact = mul_mod(fact, (n + 1) as u64, p);
        let mut coeff = mod_inv(fact, p);
        if n % 2 != 0 && coeff != 0 {
            coeff = p - coeff;
        }
        out.push(coeff);
    }
    caches.h_coeffs.insert(key, out.clone());
    out
}

fn poly_power_coeffs_mod(base: &[u64], power: i32, nmax: i32, p: u64) -> Vec<u64> {
    let size = (nmax + 1) as usize;
    let mut coeffs = vec![0_u64; size];
    coeffs[0] = 1;
    for _ in 0..power {
        let mut next = vec![0_u64; size];
        for (i, &left) in coeffs.iter().enumerate() {
            if left == 0 {
                continue;
            }
            let limit = size - i;
            for (j, &right) in base.iter().take(limit).enumerate() {
                if right != 0 {
                    let contribution = mul_mod(left, right, p);
                    add_assign_mod(&mut next[i + j], contribution, p);
                }
            }
        }
        coeffs = next;
    }
    coeffs
}

fn exp_coeffs_mod(lam_num: i32, nmax: i32, p: u64, caches: &mut Caches) -> Vec<u64> {
    let key = ExpKey { lam_num, nmax, p };
    if let Some(value) = caches.exp_coeffs.get(&key) {
        return value.clone();
    }
    let lam = mul_mod(mod_from_i128(lam_num as i128, p), mod_inv(5, p), p);
    let mut out = Vec::with_capacity((nmax + 1).max(0) as usize);
    let mut fact = 1_u64;
    let mut pow_lam = 1_u64;
    for n in 0..=nmax {
        if n != 0 {
            fact = mul_mod(fact, n as u64, p);
            pow_lam = mul_mod(pow_lam, lam, p);
        }
        out.push(mul_mod(pow_lam, mod_inv(fact, p), p));
    }
    caches.exp_coeffs.insert(key, out.clone());
    out
}

fn special_series_mod(
    power: i32,
    lam_num: i32,
    cutoff: i32,
    p: u64,
    caches: &mut Caches,
) -> Vec<(i32, u64)> {
    let nmax = cutoff + power;
    if nmax < 0 {
        return Vec::new();
    }
    if power == 0 {
        return exp_coeffs_mod(lam_num, cutoff, p, caches)
            .into_iter()
            .enumerate()
            .filter_map(|(n, coeff)| {
                if coeff != 0 {
                    Some((n as i32, coeff))
                } else {
                    None
                }
            })
            .collect();
    }
    let h = h_coeffs_mod(nmax, p, caches);
    let h_power = poly_power_coeffs_mod(&h, power, nmax, p);
    let exp_coeffs = exp_coeffs_mod(lam_num, nmax, p, caches);
    let size = (nmax + 1) as usize;
    let mut quotient = vec![0_u64; size];
    for n in 0..size {
        let mut coeff = exp_coeffs[n];
        for i in 1..=n {
            let subtract = mul_mod(h_power[i], quotient[n - i], p);
            coeff = (coeff as u128 + p as u128 - subtract as u128) as u64 % p;
        }
        quotient[n] = coeff;
    }
    quotient
        .into_iter()
        .enumerate()
        .filter_map(|(n, coeff)| {
            if coeff != 0 {
                Some((n as i32 - power, coeff))
            } else {
                None
            }
        })
        .collect()
}

fn stirling2_mod(nmax: i32, p: u64) -> Vec<Vec<u64>> {
    let size = (nmax + 1) as usize;
    let mut table = vec![vec![0_u64; size]; size];
    table[0][0] = 1;
    for n in 1..=nmax as usize {
        for k in 1..=n {
            let left = table[n - 1][k - 1];
            let right = mul_mod(k as u64, table[n - 1][k], p);
            table[n][k] = ((left as u128 + right as u128) % p as u128) as u64;
        }
    }
    table
}

fn special_derivative_dict_mod(
    order: i32,
    lam_num: i32,
    cutoff: i32,
    p: u64,
    caches: &mut Caches,
) -> Vec<(i32, u64)> {
    let key = SpecialKey {
        order,
        lam_num,
        cutoff,
        p,
    };
    if let Some(value) = caches.special.get(&key) {
        return value.clone();
    }
    let out = if order == 0 {
        special_series_mod(1, lam_num, cutoff, p, caches)
    } else {
        let mut accum: HashMap<i32, u64> = HashMap::new();
        let sign = if order % 2 == 0 { 1_u64 } else { p - 1 };
        let st = stirling2_mod(order, p);
        for k in 1..=order {
            let s2 = st[order as usize][k as usize];
            if s2 == 0 {
                continue;
            }
            let scale = mul_mod(sign, mul_mod(factorial_mod(k, p), s2, p), p);
            for (exp, coeff) in special_series_mod(k + 1, lam_num - 5 * k, cutoff, p, caches) {
                add_to_map(&mut accum, exp, mul_mod(scale, coeff, p), p);
            }
        }
        let mut items: Vec<(i32, u64)> =
            accum.into_iter().filter(|(_, coeff)| *coeff != 0).collect();
        items.sort_by_key(|(exp, _)| *exp);
        items
    };
    caches.special.insert(key, out.clone());
    out
}

fn binomial_series_mod(root_power: u32, cutoff: i32, p: u64, caches: &mut Caches) -> Vec<u64> {
    let key = BinomKey {
        root_power,
        cutoff,
        p,
    };
    if let Some(value) = caches.binomial.get(&key) {
        return value.clone();
    }
    if cutoff < 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((cutoff + 1) as usize);
    for m in 0..=cutoff as u32 {
        let mut coeff = comb_mod(root_power + m - 1, m, p);
        if m % 2 == 1 && coeff != 0 {
            coeff = p - coeff;
        }
        out.push(coeff);
    }
    caches.binomial.insert(key, out.clone());
    out
}

fn max_survivable_y_exp(
    var_idx: usize,
    deriv_orders: DerivOrders,
    denom_powers: Denom,
    current_root_pos: usize,
) -> i32 {
    let simple_pos = SIMPLE_ROOT_POS[var_idx];
    let simple_drop = if current_root_pos < simple_pos {
        denom_powers[simple_pos] as i32
    } else {
        0
    };
    deriv_orders[var_idx] + simple_drop
}

fn variable_transition_mod(
    var_idx: usize,
    deriv_orders: DerivOrders,
    y_exp: i32,
    denom_powers: Denom,
    p: u64,
    caches: &mut Caches,
) -> Vec<(Denom, u64)> {
    let key = VarKey {
        var_idx: var_idx as u8,
        deriv_orders,
        y_exp,
        denom_powers,
        p,
    };
    if let Some(value) = caches.variable.get(&key) {
        return value.clone();
    }

    let mut states: HashMap<(i32, Denom), u64> = HashMap::new();
    states.insert((y_exp, denom_powers), 1);

    for &(pos, lower_pos) in ROOT_TRANSITIONS[var_idx] {
        let mut next_states: HashMap<(i32, Denom), u64> = HashMap::new();
        for ((cur_y_exp, dtuple), state_coeff) in states.into_iter() {
            let root_power = dtuple[pos];
            if root_power == 0 {
                add_to_map(&mut next_states, (cur_y_exp, dtuple), state_coeff, p);
                continue;
            }
            let mut base_den = dtuple;
            base_den[pos] = 0;
            let y_bound = max_survivable_y_exp(var_idx, deriv_orders, base_den, pos);
            if lower_pos < 0 {
                let next_y_exp = cur_y_exp - root_power as i32;
                if next_y_exp <= y_bound {
                    add_to_map(&mut next_states, (next_y_exp, base_den), state_coeff, p);
                }
                continue;
            }
            let max_m = y_bound - cur_y_exp;
            if max_m < 0 {
                continue;
            }
            let binoms = binomial_series_mod(root_power, max_m, p, caches);
            for m in 0..=max_m {
                let transition_coeff = binoms[m as usize];
                if transition_coeff == 0 {
                    continue;
                }
                let mut expanded_den = base_den;
                expanded_den[lower_pos as usize] += root_power + m as u32;
                add_to_map(
                    &mut next_states,
                    (cur_y_exp + m, expanded_den),
                    mul_mod(state_coeff, transition_coeff, p),
                    p,
                );
            }
        }
        states = next_states
            .into_iter()
            .filter(|(_, value)| *value != 0)
            .collect();
        if states.is_empty() {
            caches.variable.insert(key, Vec::new());
            return Vec::new();
        }
    }

    let mut needed_cutoff = 0_i32;
    for ((cur_y_exp, _), _) in states.iter() {
        needed_cutoff = cmp::max(needed_cutoff, cmp::max(0, -1 - *cur_y_exp));
    }
    let special = special_derivative_dict_mod(
        deriv_orders[var_idx],
        BASE_LAMBDA_NUMS[var_idx],
        needed_cutoff,
        p,
        caches,
    );
    let special_map: HashMap<i32, u64> = special.into_iter().collect();

    let mut out: HashMap<Denom, u64> = HashMap::new();
    for ((cur_y_exp, dtuple), state_coeff) in states.into_iter() {
        if let Some(&special_coeff) = special_map.get(&(-1 - cur_y_exp)) {
            add_to_map(&mut out, dtuple, mul_mod(state_coeff, special_coeff, p), p);
        }
    }
    let mut items: Vec<(Denom, u64)> = out.into_iter().filter(|(_, coeff)| *coeff != 0).collect();
    items.sort_by_key(|(denom, _)| *denom);
    caches.variable.insert(key, items.clone());
    items
}

fn variable_transition_profile(
    var_idx: usize,
    deriv_orders: DerivOrders,
    y_exp: i32,
    denom_powers: Denom,
    p: u64,
    caches: &mut Caches,
    profile: &mut KernelProfile,
) -> Vec<(Denom, u64)> {
    profile.variable_transition_calls += 1;
    let key = VarKey {
        var_idx: var_idx as u8,
        deriv_orders,
        y_exp,
        denom_powers,
        p,
    };
    if caches.variable.contains_key(&key) {
        profile.variable_cache_hits += 1;
    } else {
        profile.variable_cache_misses += 1;
    }
    variable_transition_mod(var_idx, deriv_orders, y_exp, denom_powers, p, caches)
}

fn finish_residue_from_terms3(
    terms3: HashMap<(i32, i32, i32, Denom), u64>,
    deriv_orders: DerivOrders,
    p: u64,
    caches: &mut Caches,
) -> u64 {
    let mut terms2: HashMap<(i32, i32, Denom), u64> = HashMap::new();
    for ((a0, a1, a2, denom_powers), coeff) in terms3.into_iter() {
        for (dtuple, transition_coeff) in
            variable_transition_mod(2, deriv_orders, a2, denom_powers, p, caches)
        {
            add_to_map(
                &mut terms2,
                (a0, a1, dtuple),
                mul_mod(coeff, transition_coeff, p),
                p,
            );
        }
    }
    if terms2.is_empty() {
        return 0;
    }

    let mut terms1: HashMap<(i32, Denom), u64> = HashMap::new();
    for ((a0, a1, denom_powers), coeff) in terms2.into_iter() {
        for (dtuple, transition_coeff) in
            variable_transition_mod(1, deriv_orders, a1, denom_powers, p, caches)
        {
            add_to_map(
                &mut terms1,
                (a0, dtuple),
                mul_mod(coeff, transition_coeff, p),
                p,
            );
        }
    }
    if terms1.is_empty() {
        return 0;
    }

    let mut total = 0_u64;
    for ((a0, denom_powers), coeff) in terms1.into_iter() {
        for (dtuple, transition_coeff) in
            variable_transition_mod(0, deriv_orders, a0, denom_powers, p, caches)
        {
            if dtuple == ZERO_DENOM {
                add_assign_mod(&mut total, mul_mod(coeff, transition_coeff, p), p);
            }
        }
    }
    total
}

fn finish_residue_from_terms3_compact(
    terms3: HashMap<CompactTerms3Key, u64>,
    deriv_orders: DerivOrders,
    p: u64,
    caches: &mut Caches,
) -> u64 {
    let mut y3_items: Vec<Y3GroupedItem> = Vec::with_capacity(terms3.len());
    for ((a0, a1, a2, denom_key), coeff) in terms3.into_iter() {
        y3_items.push((a2, denom_key, a0, a1, coeff));
    }
    y3_items.sort_unstable_by_key(|item| (item.0, item.1));

    let mut terms2: HashMap<CompactTerms2Key, u64> = HashMap::new();
    let mut start = 0_usize;
    while start < y3_items.len() {
        let (a2, denom_key) = (y3_items[start].0, y3_items[start].1);
        let mut end = start + 1;
        while end < y3_items.len() && y3_items[end].0 == a2 && y3_items[end].1 == denom_key {
            end += 1;
        }
        let denom_powers = unpack_denom(denom_key);
        for (dtuple, transition_coeff) in
            variable_transition_mod(2, deriv_orders, a2, denom_powers, p, caches)
        {
            let next_denom_key = pack_denom(dtuple);
            for &(_, _, a0, a1, coeff) in y3_items[start..end].iter() {
                add_to_map(
                    &mut terms2,
                    (a0, a1, next_denom_key),
                    mul_mod(coeff, transition_coeff, p),
                    p,
                );
            }
        }
        start = end;
    }
    if terms2.is_empty() {
        return 0;
    }

    let mut y2_items: Vec<Y2GroupedItem> = Vec::with_capacity(terms2.len());
    for ((a0, a1, denom_key), coeff) in terms2.into_iter() {
        y2_items.push((a1, denom_key, a0, coeff));
    }
    y2_items.sort_unstable_by_key(|item| (item.0, item.1));

    let mut terms1: HashMap<CompactTerms1Key, u64> = HashMap::new();
    let mut start = 0_usize;
    while start < y2_items.len() {
        let (a1, denom_key) = (y2_items[start].0, y2_items[start].1);
        let mut end = start + 1;
        while end < y2_items.len() && y2_items[end].0 == a1 && y2_items[end].1 == denom_key {
            end += 1;
        }
        let denom_powers = unpack_denom(denom_key);
        for (dtuple, transition_coeff) in
            variable_transition_mod(1, deriv_orders, a1, denom_powers, p, caches)
        {
            let next_denom_key = pack_denom(dtuple);
            for &(_, _, a0, coeff) in y2_items[start..end].iter() {
                add_to_map(
                    &mut terms1,
                    (a0, next_denom_key),
                    mul_mod(coeff, transition_coeff, p),
                    p,
                );
            }
        }
        start = end;
    }
    if terms1.is_empty() {
        return 0;
    }

    let mut total = 0_u64;
    for ((a0, denom_key), coeff) in terms1.into_iter() {
        let denom_powers = unpack_denom(denom_key);
        for (dtuple, transition_coeff) in
            variable_transition_mod(0, deriv_orders, a0, denom_powers, p, caches)
        {
            if dtuple == ZERO_DENOM {
                add_assign_mod(&mut total, mul_mod(coeff, transition_coeff, p), p);
            }
        }
    }
    total
}

fn finish_residue_from_terms3_compact_profile(
    terms3: HashMap<CompactTerms3Key, u64>,
    deriv_orders: DerivOrders,
    p: u64,
    caches: &mut Caches,
    profile: &mut KernelProfile,
) -> u64 {
    profile.observe_terms3(terms3.len());

    let y3_started = Instant::now();
    let mut y3_items: Vec<Y3GroupedItem> = Vec::with_capacity(terms3.len());
    for ((a0, a1, a2, denom_key), coeff) in terms3.into_iter() {
        y3_items.push((a2, denom_key, a0, a1, coeff));
    }
    y3_items.sort_unstable_by_key(|item| (item.0, item.1));

    let mut terms2: HashMap<CompactTerms2Key, u64> = HashMap::new();
    let mut y3_group_count = 0_usize;
    let mut start = 0_usize;
    while start < y3_items.len() {
        let (a2, denom_key) = (y3_items[start].0, y3_items[start].1);
        let mut end = start + 1;
        while end < y3_items.len() && y3_items[end].0 == a2 && y3_items[end].1 == denom_key {
            end += 1;
        }
        y3_group_count += 1;
        let denom_powers = unpack_denom(denom_key);
        for (dtuple, transition_coeff) in
            variable_transition_profile(2, deriv_orders, a2, denom_powers, p, caches, profile)
        {
            let next_denom_key = pack_denom(dtuple);
            for &(_, _, a0, a1, coeff) in y3_items[start..end].iter() {
                add_to_map(
                    &mut terms2,
                    (a0, a1, next_denom_key),
                    mul_mod(coeff, transition_coeff, p),
                    p,
                );
            }
        }
        start = end;
    }
    profile.observe_y3_groups(y3_group_count);
    profile.y3_nanos += y3_started.elapsed().as_nanos();
    profile.observe_terms2(terms2.len());
    if terms2.is_empty() {
        return 0;
    }

    let y2_started = Instant::now();
    let mut y2_items: Vec<Y2GroupedItem> = Vec::with_capacity(terms2.len());
    for ((a0, a1, denom_key), coeff) in terms2.into_iter() {
        y2_items.push((a1, denom_key, a0, coeff));
    }
    y2_items.sort_unstable_by_key(|item| (item.0, item.1));

    let mut terms1: HashMap<CompactTerms1Key, u64> = HashMap::new();
    let mut y2_group_count = 0_usize;
    let mut start = 0_usize;
    while start < y2_items.len() {
        let (a1, denom_key) = (y2_items[start].0, y2_items[start].1);
        let mut end = start + 1;
        while end < y2_items.len() && y2_items[end].0 == a1 && y2_items[end].1 == denom_key {
            end += 1;
        }
        y2_group_count += 1;
        let denom_powers = unpack_denom(denom_key);
        for (dtuple, transition_coeff) in
            variable_transition_profile(1, deriv_orders, a1, denom_powers, p, caches, profile)
        {
            let next_denom_key = pack_denom(dtuple);
            for &(_, _, a0, coeff) in y2_items[start..end].iter() {
                add_to_map(
                    &mut terms1,
                    (a0, next_denom_key),
                    mul_mod(coeff, transition_coeff, p),
                    p,
                );
            }
        }
        start = end;
    }
    profile.observe_y2_groups(y2_group_count);
    profile.y2_nanos += y2_started.elapsed().as_nanos();
    profile.observe_terms1(terms1.len());
    if terms1.is_empty() {
        return 0;
    }

    let y1_started = Instant::now();
    let mut total = 0_u64;
    for ((a0, denom_key), coeff) in terms1.into_iter() {
        let denom_powers = unpack_denom(denom_key);
        for (dtuple, transition_coeff) in
            variable_transition_profile(0, deriv_orders, a0, denom_powers, p, caches, profile)
        {
            if dtuple == ZERO_DENOM {
                add_assign_mod(&mut total, mul_mod(coeff, transition_coeff, p), p);
            }
        }
    }
    profile.y1_nanos += y1_started.elapsed().as_nanos();
    total
}

fn product_terms_mod(
    left: &[(Alpha, u64)],
    right: &[(Alpha, u64)],
    p: u64,
) -> HashMap<ProductKey, u64> {
    let pair_count = left.len().saturating_mul(right.len());
    let mut products: HashMap<ProductKey, u64> = HashMap::with_capacity(pair_count);
    for &(left_alpha, left_coeff) in left.iter() {
        let left_coeff = left_coeff % p;
        if left_coeff == 0 {
            continue;
        }
        for &(right_alpha, right_coeff) in right.iter() {
            let coeff = mul_mod(left_coeff, right_coeff % p, p);
            if coeff == 0 {
                continue;
            }
            add_to_map(
                &mut products,
                (
                    left_alpha[0] + right_alpha[0],
                    left_alpha[1] + right_alpha[1],
                    left_alpha[2] + right_alpha[2],
                    left_alpha[3] + right_alpha[3],
                ),
                coeff,
                p,
            );
        }
    }
    products
}

fn residue_poly_mod_batch_impl(
    poly: &[(Alpha, u64)],
    deriv_orders: DerivOrders,
    p: u64,
    caches: &mut Caches,
) -> u64 {
    let mut terms3: HashMap<(i32, i32, i32, Denom), u64> = HashMap::new();
    for &(alpha, coeff) in poly.iter() {
        let coeff = coeff % p;
        if coeff == 0 {
            continue;
        }
        for (dtuple, transition_coeff) in
            variable_transition_mod(3, deriv_orders, alpha[3], ROOT_POWERS, p, caches)
        {
            add_to_map(
                &mut terms3,
                (alpha[0], alpha[1], alpha[2], dtuple),
                mul_mod(coeff, transition_coeff, p),
                p,
            );
        }
    }
    if terms3.is_empty() {
        return 0;
    }
    finish_residue_from_terms3(terms3, deriv_orders, p, caches)
}

fn residue_product_mod_batch_profile(
    left: &[(Alpha, u64)],
    right: &[(Alpha, u64)],
    deriv_orders: DerivOrders,
    p: u64,
    caches: &mut Caches,
    profile: &mut KernelProfile,
) -> u64 {
    profile.tasks += 1;
    profile.left_terms += left.len() as u64;
    profile.right_terms += right.len() as u64;
    profile.pair_products += (left.len() as u64) * (right.len() as u64);

    let started = Instant::now();
    let products = product_terms_mod(left, right, p);
    profile.observe_product_terms(products.len());

    let mut terms3: HashMap<CompactTerms3Key, u64> = HashMap::new();
    for ((a0, a1, a2, a3), coeff) in products.into_iter() {
        for (dtuple, transition_coeff) in
            variable_transition_profile(3, deriv_orders, a3, ROOT_POWERS, p, caches, profile)
        {
            add_to_map(
                &mut terms3,
                (a0, a1, a2, pack_denom(dtuple)),
                mul_mod(coeff, transition_coeff, p),
                p,
            );
        }
    }
    profile.product_y4_nanos += started.elapsed().as_nanos();

    if terms3.is_empty() {
        profile.observe_terms3(0);
        return 0;
    }
    finish_residue_from_terms3_compact_profile(terms3, deriv_orders, p, caches, profile)
}

fn residue_product_mod_batch_impl(
    left: &[(Alpha, u64)],
    right: &[(Alpha, u64)],
    deriv_orders: DerivOrders,
    p: u64,
    caches: &mut Caches,
) -> u64 {
    let products = product_terms_mod(left, right, p);
    if products.is_empty() {
        return 0;
    }

    let mut terms3: HashMap<CompactTerms3Key, u64> = HashMap::new();
    for ((a0, a1, a2, a3), coeff) in products.into_iter() {
        for (dtuple, transition_coeff) in
            variable_transition_mod(3, deriv_orders, a3, ROOT_POWERS, p, caches)
        {
            add_to_map(
                &mut terms3,
                (a0, a1, a2, pack_denom(dtuple)),
                mul_mod(coeff, transition_coeff, p),
                p,
            );
        }
    }
    if terms3.is_empty() {
        return 0;
    }
    finish_residue_from_terms3_compact(terms3, deriv_orders, p, caches)
}

#[no_mangle]
pub extern "C" fn rust_residue_poly_mod_batch(
    exponents: *const i32,
    coeffs: *const u64,
    len: usize,
    deriv_orders: *const i32,
    p: u64,
) -> u64 {
    if p < 3 || exponents.is_null() || coeffs.is_null() || deriv_orders.is_null() {
        return 0;
    }
    let exponents = unsafe { std::slice::from_raw_parts(exponents, len * 4) };
    let coeffs = unsafe { std::slice::from_raw_parts(coeffs, len) };
    let deriv_orders_slice = unsafe { std::slice::from_raw_parts(deriv_orders, 4) };
    let deriv_orders = [
        deriv_orders_slice[0],
        deriv_orders_slice[1],
        deriv_orders_slice[2],
        deriv_orders_slice[3],
    ];

    let mut poly = Vec::with_capacity(len);
    for idx in 0..len {
        let alpha = [
            exponents[4 * idx],
            exponents[4 * idx + 1],
            exponents[4 * idx + 2],
            exponents[4 * idx + 3],
        ];
        poly.push((alpha, coeffs[idx] % p));
    }

    let mut guard = caches().lock().expect("rust residue cache mutex poisoned");
    residue_poly_mod_batch_impl(&poly, deriv_orders, p, &mut guard)
}

#[no_mangle]
pub extern "C" fn rust_residue_product_mod_batch(
    left_exponents: *const i32,
    left_coeffs: *const u64,
    left_len: usize,
    right_exponents: *const i32,
    right_coeffs: *const u64,
    right_len: usize,
    deriv_orders: *const i32,
    p: u64,
) -> u64 {
    if p < 3
        || left_exponents.is_null()
        || left_coeffs.is_null()
        || right_exponents.is_null()
        || right_coeffs.is_null()
        || deriv_orders.is_null()
    {
        return 0;
    }

    let left_exponents = unsafe { std::slice::from_raw_parts(left_exponents, left_len * 4) };
    let left_coeffs = unsafe { std::slice::from_raw_parts(left_coeffs, left_len) };
    let right_exponents = unsafe { std::slice::from_raw_parts(right_exponents, right_len * 4) };
    let right_coeffs = unsafe { std::slice::from_raw_parts(right_coeffs, right_len) };
    let deriv_orders_slice = unsafe { std::slice::from_raw_parts(deriv_orders, 4) };
    let deriv_orders = [
        deriv_orders_slice[0],
        deriv_orders_slice[1],
        deriv_orders_slice[2],
        deriv_orders_slice[3],
    ];

    let mut left = Vec::with_capacity(left_len);
    for idx in 0..left_len {
        let alpha = [
            left_exponents[4 * idx],
            left_exponents[4 * idx + 1],
            left_exponents[4 * idx + 2],
            left_exponents[4 * idx + 3],
        ];
        left.push((alpha, left_coeffs[idx] % p));
    }

    let mut right = Vec::with_capacity(right_len);
    for idx in 0..right_len {
        let alpha = [
            right_exponents[4 * idx],
            right_exponents[4 * idx + 1],
            right_exponents[4 * idx + 2],
            right_exponents[4 * idx + 3],
        ];
        right.push((alpha, right_coeffs[idx] % p));
    }

    let mut guard = caches().lock().expect("rust residue cache mutex poisoned");
    residue_product_mod_batch_impl(&left, &right, deriv_orders, p, &mut guard)
}

#[no_mangle]
pub extern "C" fn rust_residue_products_sum_mod_batch(
    left_exponents: *const i32,
    left_coeffs: *const u64,
    left_len: usize,
    right_exponents: *const i32,
    right_coeffs: *const u64,
    right_offsets: *const usize,
    task_count: usize,
    deriv_orders: *const i32,
    p: u64,
) -> u64 {
    if p < 3
        || left_exponents.is_null()
        || left_coeffs.is_null()
        || right_exponents.is_null()
        || right_coeffs.is_null()
        || right_offsets.is_null()
        || deriv_orders.is_null()
    {
        return 0;
    }

    let left_exponents = unsafe { std::slice::from_raw_parts(left_exponents, left_len * 4) };
    let left_coeffs = unsafe { std::slice::from_raw_parts(left_coeffs, left_len) };
    let right_offsets = unsafe { std::slice::from_raw_parts(right_offsets, task_count + 1) };
    let right_total = right_offsets[task_count];
    let right_exponents = unsafe { std::slice::from_raw_parts(right_exponents, right_total * 4) };
    let right_coeffs = unsafe { std::slice::from_raw_parts(right_coeffs, right_total) };
    let deriv_orders_slice = unsafe { std::slice::from_raw_parts(deriv_orders, task_count * 4) };

    let mut left = Vec::with_capacity(left_len);
    for idx in 0..left_len {
        let alpha = [
            left_exponents[4 * idx],
            left_exponents[4 * idx + 1],
            left_exponents[4 * idx + 2],
            left_exponents[4 * idx + 3],
        ];
        left.push((alpha, left_coeffs[idx] % p));
    }

    let mut guard = caches().lock().expect("rust residue cache mutex poisoned");
    let mut total = 0_u64;
    for task_idx in 0..task_count {
        let start = right_offsets[task_idx];
        let end = right_offsets[task_idx + 1];
        if start >= end {
            continue;
        }

        let deriv_orders = [
            deriv_orders_slice[4 * task_idx],
            deriv_orders_slice[4 * task_idx + 1],
            deriv_orders_slice[4 * task_idx + 2],
            deriv_orders_slice[4 * task_idx + 3],
        ];

        let mut right = Vec::with_capacity(end - start);
        for idx in start..end {
            let alpha = [
                right_exponents[4 * idx],
                right_exponents[4 * idx + 1],
                right_exponents[4 * idx + 2],
                right_exponents[4 * idx + 3],
            ];
            right.push((alpha, right_coeffs[idx] % p));
        }

        let value = residue_product_mod_batch_impl(&left, &right, deriv_orders, p, &mut guard);
        add_assign_mod(&mut total, value, p);
    }
    total
}

#[no_mangle]
pub extern "C" fn rust_residue_products_sum_profile_json(
    left_exponents: *const i32,
    left_coeffs: *const u64,
    left_len: usize,
    right_exponents: *const i32,
    right_coeffs: *const u64,
    right_offsets: *const usize,
    task_count: usize,
    deriv_orders: *const i32,
    p: u64,
) -> *mut c_char {
    if p < 3
        || left_exponents.is_null()
        || left_coeffs.is_null()
        || right_exponents.is_null()
        || right_coeffs.is_null()
        || right_offsets.is_null()
        || deriv_orders.is_null()
    {
        return std::ptr::null_mut();
    }

    let left_exponents = unsafe { std::slice::from_raw_parts(left_exponents, left_len * 4) };
    let left_coeffs = unsafe { std::slice::from_raw_parts(left_coeffs, left_len) };
    let right_offsets = unsafe { std::slice::from_raw_parts(right_offsets, task_count + 1) };
    let right_total = right_offsets[task_count];
    let right_exponents = unsafe { std::slice::from_raw_parts(right_exponents, right_total * 4) };
    let right_coeffs = unsafe { std::slice::from_raw_parts(right_coeffs, right_total) };
    let deriv_orders_slice = unsafe { std::slice::from_raw_parts(deriv_orders, task_count * 4) };

    let mut left = Vec::with_capacity(left_len);
    for idx in 0..left_len {
        let alpha = [
            left_exponents[4 * idx],
            left_exponents[4 * idx + 1],
            left_exponents[4 * idx + 2],
            left_exponents[4 * idx + 3],
        ];
        left.push((alpha, left_coeffs[idx] % p));
    }

    let mut profile = KernelProfile::default();
    let mut guard = caches().lock().expect("rust residue cache mutex poisoned");
    let mut total = 0_u64;
    for task_idx in 0..task_count {
        let start = right_offsets[task_idx];
        let end = right_offsets[task_idx + 1];
        if start >= end {
            continue;
        }

        let deriv_orders = [
            deriv_orders_slice[4 * task_idx],
            deriv_orders_slice[4 * task_idx + 1],
            deriv_orders_slice[4 * task_idx + 2],
            deriv_orders_slice[4 * task_idx + 3],
        ];

        let right_started = Instant::now();
        let mut right = Vec::with_capacity(end - start);
        for idx in start..end {
            let alpha = [
                right_exponents[4 * idx],
                right_exponents[4 * idx + 1],
                right_exponents[4 * idx + 2],
                right_exponents[4 * idx + 3],
            ];
            right.push((alpha, right_coeffs[idx] % p));
        }
        profile.right_vec_build_nanos += right_started.elapsed().as_nanos();

        let value = residue_product_mod_batch_profile(
            &left,
            &right,
            deriv_orders,
            p,
            &mut guard,
            &mut profile,
        );
        add_assign_mod(&mut total, value, p);
    }
    profile.value = total;

    match CString::new(profile.to_json()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn rust_residue_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}

#[no_mangle]
pub extern "C" fn rust_residue_clear_caches() {
    let mut guard = caches().lock().expect("rust residue cache mutex poisoned");
    *guard = Caches::default();
}
