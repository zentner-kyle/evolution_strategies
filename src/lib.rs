//! This crate implements the Evolution Strategies (&mu;, &lambda;) and (&mu; + &lambda;).
//! There are two main types in this crate, `Problem` and `Engine`. To use a strategy, first define
//! a new implementation of `Problem`. Then create an `Engine`, and pass it an instance of the
//! problem and the desired strategy.
//!
//! The (&mu;, &lambda;) strategy (`Strategy::MuLambda`) maintains a population of &mu; (`mu`)
//! individuals, which produce &lambda; (`lambda`) offspring each generation.
//!
//! The (&mu; + &lambda;) strategy (`Strategy::MuPlusLambda`) also maintains a population of &mu;
//! (`mu`) individuals, but each generation has  &mu; + &lambda; offspring each generation, the
//! first &mu; of which are clones of the previous generation.
//!
//! In other words,
//! (&mu; + &lambda;) allows individuals from the previous generation to survive without mutating,
//! while (&mu;, &lambda;) forces the whole population to mutate each generation.
//!
//! #Example
//! ##Minimizing a quadratic function
//! ```rust
//! extern crate evolution_strategies;
//! extern crate rand;
//!
//! use evolution_strategies::{Engine, Problem, Strategy};
//! use rand::{Rng, SeedableRng, XorShiftRng};
//! use std::cmp::{Ordering, PartialOrd};
//!
//! #[derive(Clone)]
//! struct QuadraticProblem {
//!     a: f64,
//!     b: f64,
//!     c: f64,
//!     min: f64,
//!     max: f64,
//!     epsilon: f64,
//! }
//!
//! impl QuadraticProblem {
//!     fn value(&self, x: f64) -> f64 {
//!         return self.a * x * x + self.b * x + self.c;
//!     }
//! }
//!
//! impl Problem for QuadraticProblem {
//!     type Individual = f64;
//!
//!     fn initialize<R>(&mut self, count: usize, rng: &mut R) -> Vec<Self::Individual>
//!         where R: Rng
//!     {
//!         (0..count)
//!             .map(|_| rng.gen_range(self.min, self.max))
//!             .collect()
//!     }
//!
//!     fn mutate<R>(&mut self, individual: &mut Self::Individual, rng: &mut R) -> bool
//!         where R: Rng
//!     {
//!         *individual += rng.next_f64() * self.epsilon;
//!         return true;
//!     }
//!
//!     fn compare<R>(&mut self,
//!                   a: &Self::Individual,
//!                   b: &Self::Individual,
//!                   _rng: &mut R)
//!                   -> Option<Ordering>
//!         where R: Rng
//!     {
//!         self.value(*b).partial_cmp(&self.value(*a))
//!     }
//! }
//!
//! fn main() {
//!     let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
//!     let problem = QuadraticProblem {
//!         a: 2.0,
//!         b: 0.0,
//!         c: 0.0,
//!         min: -1.0,
//!         max: 1.0,
//!         epsilon: 0.01,
//!     };
//!     let strategy = Strategy::MuPlusLambda { mu: 10, lambda: 5 };
//!     let mut engine = Engine::new(problem, strategy, rng);
//!     for i in 0..1000 {
//!         if i % 100 == 0 {
//!             let fitest = *engine.fitest();
//!             println!("fitest = {}", fitest);
//!             println!("value(fitest) = {}", engine.problem().value(fitest));
//!         }
//!         engine.run_generation();
//!     }
//!     assert!(engine.problem().value(*engine.fitest()).abs() < 1e-5);
//! }
//! ```
extern crate rand;

use rand::Rng;
use std::{cmp, mem, slice};
use std::cmp::Ordering;


/// An instance of a problem to be solved using an evolutionary strategy.
pub trait Problem {
    type Individual: CloneMut;

    /// Create an initial population of `count` individuals.
    fn initialize<R>(&mut self, count: usize, rng: &mut R) -> Vec<Self::Individual> where R: Rng;


    /// Mutate an individual. Should return true if the individual's fitness may have changed.
    fn mutate<R>(&mut self, &mut Self::Individual, rng: &mut R) -> bool where R: Rng;


    /// Compare two individuals by fitness.
    /// If None is returned, the `Engine` will choose a more fit individual randomly.
    fn compare<R>(&mut self,
                  a: &Self::Individual,
                  b: &Self::Individual,
                  rng: &mut R)
                  -> Option<cmp::Ordering>
        where R: Rng;


    /// Checks if an individual is optimal.
    fn is_optimal(&mut self, _individual: &Self::Individual) -> bool {
        return false;
    }


    /// Run some maintenance on the `Problem` and population.
    fn maintain<R>(&mut self, _population: Population<Self::Individual>, _rng: &mut R) where R: Rng {
    }
}

/// (&mu;, &lambda;) or (&mu; + &lambda;)
#[derive(Clone, Copy)]
pub enum Strategy {
    /// (&mu;, &lambda;)
    MuLambda { mu: usize, lambda: usize },
    /// (&mu; + &lambda;)
    MuPlusLambda { mu: usize, lambda: usize },
}

impl Strategy {
    fn population(self) -> usize {
        match self {
            Strategy::MuLambda { mu, .. } => mu,
            Strategy::MuPlusLambda { mu, .. } => mu,
        }
    }
}

/// Implements the evolution strategies, and allows stepping through generations.
pub struct Engine<P: Problem, R: Rng> {
    strategy: Strategy,
    population: Vec<P::Individual>,
    previous_generation: Vec<P::Individual>,
    problem: P,
    rng: R,
    needs_sort: bool,
}

/// Iterator over an `Engine`'s population.
/// Returned by `Engine::population()`.
pub struct PopulationIter<'a, I: 'a> {
    slice_iter: slice::Iter<'a, I>,
}

impl<'a, I: 'a> Clone for PopulationIter<'a, I> {
    fn clone(&self) -> Self {
        PopulationIter { slice_iter: self.slice_iter.as_slice().iter() }
    }
}

impl<'a, I: 'a> Iterator for PopulationIter<'a, I> {
    type Item = &'a I;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice_iter.next()
    }
}

/// Iterator over an `Engine`'s population.
/// Returned by `Engine::population_mut()`.
pub struct PopulationIterMut<'a, I: 'a> {
    slice_iter: slice::IterMut<'a, I>,
}

impl<'a, I: 'a> PopulationIterMut<'a, I> {
    pub fn unmut(self) -> PopulationIter<'a, I> {
        PopulationIter { slice_iter: self.slice_iter.into_slice().iter() }
    }
}

impl<'a, I: 'a> Iterator for PopulationIterMut<'a, I> {
    type Item = &'a mut I;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice_iter.next()
    }
}

pub struct Population<'a, I: 'a> {
    slice: &'a mut [I],
}

impl<'a, I: 'a> Population<'a, I> {
    pub fn iter_mut<'b>(&'b mut self) -> PopulationIterMut<'b, I> {
        PopulationIterMut { slice_iter: self.slice.iter_mut() }
    }

    pub fn iter<'b>(&'b self) -> PopulationIter<'b, I> {
        PopulationIter { slice_iter: self.slice.iter() }
    }
}

impl<P: Problem, R: Rng> Engine<P, R> {
    pub fn new(mut problem: P, strategy: Strategy, mut rng: R) -> Self {
        let mut engine = Engine {
            strategy,
            population: problem.initialize(strategy.population(), &mut rng),
            previous_generation: Vec::new(),
            problem,
            rng,
            needs_sort: true,
        };
        engine.reduce_population();
        engine
    }

    /// Randomly choose `offspring_count` parents (with replacement), and mutate them to produce
    /// offspring.
    fn produce_offspring(&mut self, offspring_count: usize) {
        for _ in 0..offspring_count {
            let parent = self.rng
                .choose_mut(&mut self.previous_generation)
                .unwrap();
            let mut child = parent.clone_mut();
            self.needs_sort |= self.problem.mutate(&mut child, &mut self.rng);
            self.population.push(child);
        }
    }

    /// Copy the previous generation into the population.
    fn copy_previous_generation(&mut self) {
        self.population = self.previous_generation
            .iter_mut()
            .map(|i| i.clone_mut())
            .collect();
    }

    /// Reduce the population size to that required by the strategy.
    fn reduce_population(&mut self) {
        let rng_ref = &mut self.rng;
        let problem_ref = &mut self.problem;
        if self.needs_sort {
            self.population
                .sort_by(|a, b| match problem_ref.compare(b, a, rng_ref) {
                             Some(ordering) => ordering,
                             None => {
                                 if rng_ref.gen() {
                                     Ordering::Less
                                 } else {
                                     Ordering::Greater
                                 }
                             }
                         });
            self.needs_sort = false;
        }
        self.population.truncate(self.strategy.population());
    }

    pub fn run_generation(&mut self) {
        mem::swap(&mut self.previous_generation, &mut self.population);
        self.population.clear();
        match self.strategy {
            Strategy::MuLambda { lambda, .. } => {
                self.produce_offspring(lambda);
                self.reduce_population();
            }
            Strategy::MuPlusLambda { lambda, .. } => {
                self.copy_previous_generation();
                self.produce_offspring(lambda);
                self.reduce_population();
            }
        }
    }

    /// Call maintain on the Problem until it is done, passing in the `Engine`'s population and rng.
    pub fn maintain(&mut self) {
        let population = Population { slice: &mut self.population };
        self.problem.maintain(population, &mut self.rng);
    }

    /// Searches the population for an optimal individual.
    /// Take O(|population|) time.
    pub fn optimal(&mut self) -> Option<P::Individual> {
        for individual in &mut self.population {
            if self.problem.is_optimal(individual) {
                return Some(individual.clone_mut());
            }
        }
        return None;
    }

    /// Retrieve the most fit individual.
    /// Takes O(1) time.
    pub fn fitest(&self) -> &P::Individual {
        &self.population[0]
    }

    /// Iterate over individuals from most fit to least fit.
    pub fn population(&self) -> PopulationIter<P::Individual> {
        PopulationIter { slice_iter: self.population.iter() }
    }

    /// Iterate over individuals from most fit to least fit.
    pub fn population_mut(&mut self) -> PopulationIterMut<P::Individual> {
        PopulationIterMut { slice_iter: self.population.iter_mut() }
    }

    /// Access the Problem passed to the Engine.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// Mutably access the Problem passed to the Engine.
    pub fn mut_problem(&mut self) -> &mut P {
        &mut self.problem
    }
}

/// Similar trait to Clone, but taking a mutable reference instead.
pub trait CloneMut {
    fn clone_mut(&mut self) -> Self;
}

impl<T> CloneMut for T
    where T: Clone
{
    fn clone_mut(&mut self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::XorShiftRng;
    use std::cmp::PartialOrd;

    #[derive(Clone)]
    struct QuadraticProblem {
        a: f64,
        b: f64,
        c: f64,
        min: f64,
        max: f64,
        epsilon: f64,
    }

    impl QuadraticProblem {
        fn value(&self, x: f64) -> f64 {
            return self.a * x * x + self.b * x + self.c;
        }
    }

    impl Problem for QuadraticProblem {
        type Individual = f64;

        fn initialize<R>(&mut self, count: usize, rng: &mut R) -> Vec<Self::Individual>
            where R: Rng
        {
            (0..count)
                .map(|_| rng.gen_range(self.min, self.max))
                .collect()
        }

        fn mutate<R>(&mut self, individual: &mut Self::Individual, rng: &mut R) -> bool
            where R: Rng
        {
            *individual += rng.next_f64() * self.epsilon;
            return true;
        }

        fn compare<R>(&mut self,
                      a: &Self::Individual,
                      b: &Self::Individual,
                      _rng: &mut R)
                      -> Option<Ordering>
            where R: Rng
        {
            self.value(*b).partial_cmp(&self.value(*a))
        }
    }

    #[test]
    fn solve_quadratic() {
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let problem = QuadraticProblem {
            a: 2.0,
            b: 0.0,
            c: 0.0,
            min: -1.0,
            max: 1.0,
            epsilon: 0.01,
        };
        let strategy = Strategy::MuPlusLambda { mu: 10, lambda: 5 };
        let mut engine = Engine::new(problem, strategy, rng);
        for i in 0..1000 {
            if i % 100 == 0 {
                let fitest = *engine.fitest();
                println!("fitest = {}", fitest);
                println!("value(fitest) = {}", engine.problem().value(fitest));
            }
            engine.run_generation();
        }
        assert!(engine.problem().value(*engine.fitest()).abs() < 1e-5);
    }
}
