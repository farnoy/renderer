use std::fmt::Debug;

use derive_syn_parse::Parse;
use proc_macro2::TokenTree;
use quote::quote;
use serde::{Deserialize, Serialize};
use syn::{
    braced, bracketed,
    parse::{Parse, ParseStream, Parser, Result},
    parse_quote,
    punctuated::Punctuated,
    token::{Brace, Bracket},
    Expr, Field, Ident, LitBool, LitInt, Path, Token, Type,
};

use super::keywords as kw;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DependencyType {
    SameFrame,
    LastFrame,
    LastAccess,
}

impl Parse for DependencyType {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        bracketed!(inner in input);
        inner
            .parse::<kw::same_frame>()
            .and(Ok(DependencyType::SameFrame))
            .or(inner.parse::<kw::last_frame>().and(Ok(DependencyType::LastFrame)))
            .or(inner.parse::<kw::last_access>().and(Ok(DependencyType::LastAccess)))
    }
}

#[derive(Clone)]
pub struct Sequence<A, B>(pub (A, B));

impl<A: Parse, B: Parse> Parse for Sequence<A, B> {
    fn parse(input: ParseStream) -> Result<Self> {
        // Forking here so that Sequence only consumes anything if it can parse everything
        let peeking = input.fork();
        let a = peeking.parse::<A>();
        let b = peeking.parse::<B>();

        if a.is_ok() && b.is_ok() {
            input
                .parse()
                .and_then(|a| input.parse().map(|b| (a, b)))
                .map(|tup| Sequence(tup))
        } else {
            a.and_then(|a| b.map(|b| (a, b))).map(|tup| Sequence(tup))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Unbrace<T>(pub T);

impl<T: Parse> Parse for Unbrace<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        braced!(inner in input);
        let t = inner.parse()?;
        Ok(Unbrace(t))
    }
}

#[derive(Clone, Debug)]
pub struct Unbracket<T>(pub T);

impl<T: Parse> Parse for Unbracket<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        bracketed!(inner in input);
        let t = inner.parse()?;
        Ok(Unbracket(t))
    }
}

#[derive(Debug, Clone)]
pub struct UnArray<T>(pub Vec<T>);

impl<T: Parse> Parse for UnArray<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(UnArray(
            Punctuated::<T, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct UnOption<T>(pub Option<T>);

impl<T: Parse> Parse for UnOption<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<T>().map(|t| UnOption(Some(t))).or(Ok(UnOption(None)))
    }
}

#[derive(Clone)]
pub struct ArrowPair<A, B>(pub (A, B));

impl<A: Parse, B: Parse> Parse for ArrowPair<A, B> {
    fn parse(input: ParseStream) -> Result<Self> {
        let a = input.parse()?;
        input.parse::<Token![=>]>()?;
        let b = input.parse()?;
        Ok(ArrowPair((a, b)))
    }
}

pub fn to_vk_format(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let parser = |input: ParseStream| {
        input
            .parse::<kw::vec2>()
            .and(Ok(quote!(vk::Format::R32G32_SFLOAT)))
            .or_else(|_| input.parse::<kw::vec3>().and(Ok(quote!(vk::Format::R32G32B32_SFLOAT))))
            .or_else(|_| {
                input
                    .parse::<kw::vec4>()
                    .and(Ok(quote!(vk::Format::R32G32B32A32_SFLOAT)))
            })
    };

    parser.parse2(input).unwrap()
}

pub fn to_rust_type(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let parser = |input: ParseStream| {
        input
            .parse::<kw::vec2>()
            .and(Ok(quote!(glm::Vec2)))
            .or_else(|_| input.parse::<kw::vec3>().and(Ok(quote!(glm::Vec3))))
            .or_else(|_| input.parse::<kw::vec4>().and(Ok(quote!(glm::Vec4))))
            .or_else(|_| input.parse::<TokenTree>().map(|t| quote!(#t)))
    };

    parser.parse2(input).unwrap()
}
