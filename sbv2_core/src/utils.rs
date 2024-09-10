pub fn intersperse<T>(slice: &[T], sep: T) -> Vec<T>
where
    T: Clone,
{
    /*
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
    */
    let mut result = vec![sep.clone(); slice.len() * 2 + 1];
    result
        .iter_mut()
        .step_by(2)
        .zip(slice.iter())
        .for_each(|(r, s)| *r = s.clone());
    result
}
