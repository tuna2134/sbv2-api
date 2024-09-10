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

/*
fn tile<T: Clone>(arr: &Array2<T>, reps: (usize, usize)) -> Array2<T> {
    let (rows, cols) = arr.dim();
    let (rep_rows, rep_cols) = reps;

    let mut result = Array::zeros((rows * rep_rows, cols * rep_cols));

    for i in 0..rep_rows {
        for j in 0..rep_cols {
            let view = result.slice_mut(s![
                i * rows..(i + 1) * rows,
                j * cols..(j + 1) * cols
            ]);
            view.assign(arr);
        }
    }

    result
}
*/
