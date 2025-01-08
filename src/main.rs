use std::{iter::Peekable, str::Chars};

use miette::IntoDiagnostic;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(std::rc::Rc<String>);

impl Name {
  pub fn new(s: impl Into<String>) -> Self {
    Self(std::rc::Rc::from(s.into()))
  }
}

impl PartialEq<str> for Name {
  fn eq(&self, other: &str) -> bool {
    self.0.as_ref() == other
  }
}

impl std::fmt::Display for Name {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Text(std::rc::Rc<String>);

#[derive(Clone, Debug)]
pub enum ExprStruct {
  Atom(Name),
  Ident(Name),
  String(Text),
  Number(f64),
  List(Vec<Expr>),
  Vector(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct Expr {
  pub expr: ExprStruct,
  pub loc: Loc,
}

#[derive(Clone, Debug)]
pub struct Loc {
  pub start: usize,
  pub end: usize,
  pub source: Source,
}

impl From<Loc> for miette::SourceSpan {
  fn from(value: Loc) -> Self {
    let length = value.end - value.start;
    Self::new(miette::SourceOffset::from(value.start), length)
  }
}

#[derive(Clone, Debug)]
pub struct Source {
  source: miette::NamedSource<std::sync::Arc<String>>,
}

impl Source {
  pub fn text(&self) -> &str {
    self.source.inner()
  }

  pub fn get_text(&self) -> String {
    self.source.inner().to_string()
  }
}

macro_rules! impl_source_for {
  ($t:ident) => {
    impl miette::SourceCode for $t {
      fn read_span<'a>(
        &'a self,
        span: &miette::SourceSpan,
        context_lines_before: usize,
        context_lines_after: usize,
      ) -> Result<Box<dyn miette::SpanContents<'a> + 'a>, miette::MietteError> {
        self
          .source
          .read_span(span, context_lines_before, context_lines_after)
      }
    }
  };
}

impl_source_for!(Loc);
impl_source_for!(Source);

impl TryFrom<std::path::PathBuf> for Source {
  type Error = miette::Report;

  fn try_from(path: std::path::PathBuf) -> Result<Self, Self::Error> {
    let text = std::fs::read_to_string(&path).into_diagnostic()?;
    let source = miette::NamedSource::new(path.to_str().unwrap_or(""), std::sync::Arc::new(text))
      .with_language("candy");
    Ok(Self { source })
  }
}

pub fn parse_from_source(source: Source) -> miette::Result<Vec<Expr>> {
  let src = source.get_text().clone();
  let lexer = Lexer {
    source,
    peekable: src.chars().peekable(),
    src: &src,
    start: 0,
    index: 0,
  };
  let mut parser = Parser::new(lexer);
  parser.exprs()
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenStruct {
  Ident(Name),
  Atom(Name),
  String(Text),
  Number(f64),
  LParens,
  RParens,
  LBracket,
  RBracket,
  Quote,
  Unquote,
  Error,
  Eof,
}

impl std::fmt::Display for TokenStruct {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      TokenStruct::Ident(i) => write!(f, "{i}"),
      TokenStruct::Atom(i) => write!(f, ":{i}"),
      TokenStruct::String(s) => write!(f, "{s:?}"),
      TokenStruct::Number(n) => write!(f, "{n}"),
      TokenStruct::LParens => write!(f, "("),
      TokenStruct::RParens => write!(f, ")"),
      TokenStruct::LBracket => write!(f, "["),
      TokenStruct::RBracket => write!(f, "]"),
      TokenStruct::Quote => write!(f, "'"),
      TokenStruct::Unquote => write!(f, "~"),
      TokenStruct::Error => write!(f, "error"),
      TokenStruct::Eof => write!(f, "EOF"),
    }
  }
}

#[derive(Clone, Debug)]
pub struct Token {
  token: TokenStruct,
  loc: Loc,
}

pub struct Lexer<'source> {
  source: Source,
  peekable: Peekable<Chars<'source>>,
  src: &'source str,
  start: usize,
  index: usize,
}

impl<'source> Lexer<'source> {
  const RESTRICTED: &'static str = " \n\r\t()[]'~";

  fn save(&mut self) {
    self.start = self.index;
  }

  fn advance(&mut self) -> Option<char> {
    let char = self.peekable.next()?;
    self.index += char.len_utf8();
    Some(char)
  }

  fn advance_while(&mut self, cond: impl Fn(&char) -> bool) {
    while let Some(char) = self.peekable.peek() {
      if cond(char) {
        self.advance();
      } else {
        break;
      }
    }
  }

  fn skip(&mut self) {
    while let Some(char) = self.peekable.peek() {
      match char {
        char if char.is_ascii_whitespace() => _ = self.advance(),
        ';' => {
          self.advance_while(|c| *c != '\n');
          continue;
        }
        _ => break,
      }
    }
  }

  fn contents(&self) -> String {
    self.src[self.start..self.index].to_owned()
  }

  fn read_string(&mut self) -> TokenStruct {
    let mut buf = String::with_capacity(16);
    while let Some(char) = self.peekable.peek() {
      if *char == '"' {
        break;
      }
      buf.push(self.advance().unwrap());
    }
    match self.advance() {
      Some('"') => TokenStruct::String(Text(buf.into())),
      _ => TokenStruct::Error,
    }
  }

  fn read_atom(&mut self) -> TokenStruct {
    self.save();
    self.advance_while(|c| !Self::RESTRICTED.contains(*c));
    TokenStruct::Atom(Name(self.contents().into()))
  }

  pub fn token(&mut self) -> TokenStruct {
    self.skip();
    self.save();
    if let Some(char) = self.advance() {
      match char {
        '(' => TokenStruct::LParens,
        ')' => TokenStruct::RParens,
        '[' => TokenStruct::LBracket,
        ']' => TokenStruct::RBracket,
        ':' => self.read_atom(),
        // TODO: splice-unquote
        '\'' => TokenStruct::Quote,
        '~' => TokenStruct::Unquote,
        '"' => self.read_string(),
        '-' | '+' => match self.peekable.peek() {
          Some(c) if c.is_ascii_digit() => {
            self.advance_while(|c| c.is_ascii_digit());
            if let Some('.') = self.peekable.peek() {
              self.advance();
              if let Some(c) = self.peekable.peek() {
                if c.is_ascii_digit() {
                  self.advance_while(|c| c.is_ascii_digit());
                }
              }
            }
            TokenStruct::Number(self.contents().parse().unwrap())
          }
          _ => {
            self.advance_while(|c| !Self::RESTRICTED.contains(*c));
            TokenStruct::Ident(Name(self.contents().into()))
          }
        },
        c if c.is_ascii_digit() => {
          self.advance_while(|c| c.is_ascii_digit());
          if let Some('.') = self.peekable.peek() {
            self.advance();
            self.advance_while(|c| c.is_ascii_digit());
          }
          TokenStruct::Number(self.contents().parse().unwrap())
        }
        c if !Self::RESTRICTED.contains(c) => {
          self.advance_while(|c| !Self::RESTRICTED.contains(*c));
          TokenStruct::Ident(Name(std::rc::Rc::new(self.contents())))
        }
        _ => TokenStruct::Error,
      }
    } else {
      TokenStruct::Eof
    }
  }

  fn loc(&self) -> Loc {
    Loc {
      start: self.start,
      end: self.index,
      source: self.source.clone(),
    }
  }

  pub fn next_token(&mut self) -> Token {
    let token = self.token();
    let loc = self.loc();
    Token { token, loc }
  }
}

pub enum OpenType {
  Parens,
  Bracket,
}

impl OpenType {
  fn to_parser_error(self, loc: Loc) -> ParserError {
    match self {
      OpenType::Parens => ParserError::UnclosedParens(loc),
      OpenType::Bracket => ParserError::UnclosedBracket(loc),
    }
  }
}

pub struct Parser<'source> {
  opens: Vec<(Loc, OpenType)>,
  lexer: Lexer<'source>,
  curr: Token,
  next: Token,
}

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
#[diagnostic()]
pub enum ParserError {
  #[error("unexpected token")]
  Unexpected(
    #[label("here")]
    #[source_code]
    Loc,
  ),
  #[error("unclosed parens")]
  UnclosedParens(
    #[label("here")]
    #[source_code]
    Loc,
  ),
  #[error("unclosed bracket")]
  UnclosedBracket(
    #[label("here")]
    #[source_code]
    Loc,
  ),
  #[error("reached eof")]
  ReachedEof,
}

impl<'source> Parser<'source> {
  pub fn new(mut lexer: Lexer<'source>) -> Self {
    Self {
      opens: Vec::new(),
      curr: lexer.next_token(),
      next: lexer.next_token(),
      lexer,
    }
  }

  fn pop_open_loc(&mut self) -> Loc {
    self.opens.pop().unwrap().0
  }

  fn advance(&mut self) -> Token {
    let next = self.lexer.next_token();
    let next = std::mem::replace(&mut self.next, next);
    std::mem::replace(&mut self.curr, next)
  }

  fn expr(&mut self) -> miette::Result<Expr> {
    let token = self.advance();
    let (expr, loc) = match token.token {
      TokenStruct::Ident(i) => (ExprStruct::Ident(i), token.loc),
      TokenStruct::Atom(i) => (ExprStruct::Atom(i), token.loc),
      TokenStruct::String(s) => (ExprStruct::String(s), token.loc),
      TokenStruct::Number(n) => (ExprStruct::Number(n), token.loc),
      TokenStruct::LParens => {
        let start = token.loc;
        self.opens.push((start.clone(), OpenType::Parens));
        let mut exprs = Vec::new();
        let end;

        loop {
          if self.curr.token == TokenStruct::Eof {
            Err(ParserError::UnclosedParens(self.pop_open_loc()))?
          }
          if self.curr.token == TokenStruct::RParens {
            end = self.curr.loc.clone();
            self.advance();
            self.opens.pop().unwrap();
            break;
          }
          exprs.push(self.expr()?);
        }
        (
          ExprStruct::List(exprs),
          Loc {
            start: start.start,
            end: end.end,
            source: start.source,
          },
        )
      }
      TokenStruct::LBracket => {
        let start = token.loc;
        self.opens.push((start.clone(), OpenType::Bracket));
        let mut exprs = Vec::new();
        let end;

        loop {
          if self.curr.token == TokenStruct::Eof {
            Err(ParserError::UnclosedBracket(self.pop_open_loc()))?
          }
          if self.curr.token == TokenStruct::RBracket {
            end = self.curr.loc.clone();
            self.advance();
            self.opens.pop().unwrap();
            break;
          }
          exprs.push(self.expr()?);
        }
        (
          ExprStruct::Vector(exprs),
          Loc {
            start: start.start,
            end: end.end,
            source: start.source,
          },
        )
      }
      TokenStruct::Quote | TokenStruct::Unquote => {
        let expr = self.expr()?;
        let expr_loc = expr.loc.clone();
        let name = match token.token {
          TokenStruct::Quote => "quote",
          TokenStruct::Unquote => "unquote",
          _ => unreachable!(),
        };
        let quote_expr = Expr {
          expr: ExprStruct::Ident(Name(name.to_string().into())),
          loc: token.loc,
        };
        (ExprStruct::List(vec![quote_expr, expr]), expr_loc)
      }
      TokenStruct::RParens | TokenStruct::RBracket => {
        if !self.opens.is_empty() {
          let (loc, open_type) = self.opens.pop().unwrap();
          Err(open_type.to_parser_error(loc))?
        }
        Err(ParserError::Unexpected(token.loc))?
      }
      TokenStruct::Error => Err(ParserError::Unexpected(token.loc))?,
      TokenStruct::Eof => Err(ParserError::ReachedEof)?,
    };

    Ok(Expr { expr, loc })
  }

  fn exprs(&mut self) -> miette::Result<Vec<Expr>> {
    let mut exprs = Vec::new();
    while self.curr.token != TokenStruct::Eof {
      exprs.push(self.expr()?);
    }
    Ok(exprs)
  }
}

impl ExprStruct {
  pub fn split(self) -> Option<(Expr, Vec<Expr>)> {
    if let ExprStruct::List(list) = self {
      let (head, tail) = list.split_first()?;
      Some((head.clone(), tail.to_vec()))
    } else {
      None
    }
  }

  pub fn name(self) -> Option<Name> {
    if let ExprStruct::Ident(name) = self {
      Some(name)
    } else {
      None
    }
  }

  pub fn is_ident(&self, name: &str) -> bool {
    matches!(self, ExprStruct::Ident(name_) if name_ == name)
  }

  pub fn elements(&self) -> Option<Vec<Expr>> {
    if let ExprStruct::List(elements) | ExprStruct::Vector(elements) = self {
      Some(elements.clone())
    } else {
      None
    }
  }
}

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
  #[arg()]
  main: std::path::PathBuf,
}

#[derive(Clone, Default)]
pub struct EnvStruct {
  map: std::cell::RefCell<std::collections::HashMap<Name, Value>>,
  parent: Option<Env>,
}

impl EnvStruct {
  pub fn from(outer: &Env) -> Env {
    Env::new(EnvStruct {
      map: Default::default(),
      parent: Some(outer.clone()),
    })
  }

  pub fn fetch(self: &std::rc::Rc<Self>, name: &Name) -> Option<Value> {
    match self.map.borrow().get(name) {
      Some(def) => Some(def.clone()),
      None => {
        if let Some(outer) = self.parent.clone() {
          outer.fetch(name)
        } else {
          None
        }
      }
    }
  }

  pub fn define(&self, name: Name, value: Value) {
    self.map.borrow_mut().insert(name, value);
  }
}

type Env = std::rc::Rc<EnvStruct>;
impl std::fmt::Debug for EnvStruct {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "<env>")
  }
}

#[derive(Clone, Debug, Default)]
pub enum Value {
  Number(f64),
  String(Text),
  Symbol(Name),
  Atom(Name),
  Fun(Fun),
  Macro(Fun),
  List(Vec<Value>),
  #[default]
  Nil,
}

impl std::fmt::Display for Value {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Value::Number(num) => write!(f, "{num}"),
      Value::String(Text(text)) => write!(f, "{text}"),
      Value::Symbol(name) => write!(f, "{name}"),
      Value::Atom(atom) => write!(f, ":{atom}"),
      Value::Fun(_) => write!(f, "<fun>"),
      Value::Macro(_) => write!(f, "<macro>"),
      Value::List(list) => {
        let s = list
          .iter()
          .map(|e| e.to_string())
          .collect::<Vec<String>>()
          .join(" ");
        write!(f, "({s})")
      }
      Value::Nil => write!(f, "nil"),
    }
  }
}

#[derive(Clone, Debug)]
pub struct Fun {
  parameters: Vec<Name>,
  body: Box<Value>,
  env: Env,
}

macro_rules! symbol {
  ($name:expr) => {
    Value::Symbol(Name::new($name))
  };
}

macro_rules! list {
  ($($args:expr),*) => {
    Value::List(vec![$($args),*])
  }
}

impl Fun {
  fn call(&self, arguments: Vec<Value>) -> Result<Value, Value> {
    if arguments.len() != self.parameters.len() {
      Err(symbol!("error/arity-error"))?
    }

    let pairs = self.parameters.clone().into_iter().zip(arguments);

    let ref env = self.env.clone();
    for (name, value) in pairs {
      env.define(name, value);
    }

    self.body.clone().eval(env)
  }
}

impl From<Expr> for Value {
  fn from(value: Expr) -> Self {
    match value.expr {
      ExprStruct::Atom(atom) => Value::Atom(atom),
      ExprStruct::Ident(ref name) if name == "nil" => Value::Nil,
      ExprStruct::Ident(name) => Value::Symbol(name),
      ExprStruct::String(str) => Value::String(str),
      ExprStruct::Number(num) => Value::Number(num),
      ExprStruct::List(list) => Value::List(list.into_iter().map(Value::from).collect()),
      ExprStruct::Vector(list) => list![
        symbol!("quote"),
        Value::List(list.into_iter().map(Value::from).collect())
      ],
    }
  }
}

impl Value {
  fn name(self) -> Result<Name, Value> {
    if let Value::Symbol(symbol) = self {
      Ok(symbol)
    } else {
      Err(list![symbol!("error/is-not-a-symbol"), self])
    }
  }

  fn list(self) -> Result<Vec<Value>, Value> {
    match self {
      Value::List(list) => Ok(list),
      Value::Nil => Ok(vec![]),
      _ => Err(list![symbol!("error/is-not-a-list"), self]),
    }
  }

  fn symbol_list(self) -> Result<Vec<Name>, Value> {
    match self {
      Value::List(list) => list.into_iter().map(|e| e.name()).collect::<Result<_, _>>(),
      Value::Nil => Ok(vec![]),
      _ => Err(list![symbol!("error/is-not-a-symbol-list"), self]),
    }
  }

  fn splice_unquote(list: Vec<Value>) -> Result<Value, Value> {
    let mut result = list![];

    for value in list.into_iter().rev() {
      if let Value::List(ref list) = value {
        if list.len() == 2 {
          if let Value::Symbol(ref name) = list[0] {
            if name == "splice-unquote" {
              result = list![symbol!("concat"), list[1].clone(), result];
              continue;
            }
          }
        }
      }
      result = list![symbol!("cons"), value.quasiquote()?, result];
    }

    Ok(result)
  }

  fn quasiquote(self) -> Result<Self, Self> {
    match self {
      Value::List(list) if !list.is_empty() => {
        if let Some((head, tail)) = list.split_first() {
          match head {
            Value::Symbol(ref name) if name == "unquote" => {
              return tail.get(0).cloned().ok_or(symbol!("error/expected-expr"));
            }
            _ => Self::splice_unquote(list),
          }
        } else {
          Self::splice_unquote(list)
        }
      }
      Value::Symbol(name) => Ok(list![symbol!("quote"), Value::Symbol(name)]),
      _ => Ok(self),
    }
  }

  fn apply(mut self, env: &Env) -> Result<Value, Value> {
    loop {
      match self {
        Value::List(list) => {
          let (head, tail) = list.split_first().unwrap();

          match head {
            Value::Symbol(ref name) if name == "quote" => {
              return Ok(tail.get(0).ok_or(symbol!("error/expected-expr"))?.clone());
            }
            Value::Symbol(ref name) if name == "quasiquote" => {
              let expr = tail.get(0).cloned().ok_or(symbol!("error/expected-expr"))?;
              self = expr.quasiquote()?;
              continue;
            }
            Value::Symbol(ref name) if name == "progn" => {
              let mut result = Value::Nil;
              for value in tail.iter().cloned() {
                result = value.eval(env)?;
              }
              break Ok(result);
            }
            Value::Symbol(ref name) if name == "print" => {
              let expr = tail
                .get(0)
                .cloned()
                .ok_or(symbol!("error/expected-expr"))?
                .eval(env)?;
              println!("{expr}");
              break Ok(Value::Nil);
            }
            Value::Symbol(ref name) if name == "if" => {
              let cond = tail.get(0).cloned().ok_or(symbol!("error/expected-expr"))?;
              let cond_then = tail.get(1).cloned().ok_or(symbol!("error/expected-expr"))?;
              let cond_else = tail.get(2).cloned().ok_or(symbol!("error/expected-expr"))?;

              let branch = match cond.eval(env)? {
                Value::Nil => cond_else,
                _ => cond_then,
              };

              break branch.eval(env);
            }
            Value::Symbol(ref name) if name == "cons" => {
              let a = tail
                .get(0)
                .cloned()
                .ok_or(symbol!("error/expected-expr"))?
                .eval(env)?;
              let b = tail
                .get(1)
                .cloned()
                .ok_or(symbol!("error/expected-expr"))?
                .eval(env)?
                .list()?;
              let mut a = vec![a];
              a.extend(b);
              return Ok(Value::List(a));
            }
            Value::Symbol(ref name) if name == "list" => {
              return Ok(Value::List(
                tail
                  .iter()
                  .cloned()
                  .map(|e| e.eval(env))
                  .collect::<Result<_, _>>()?,
              ))
            }
            Value::Symbol(ref name) if name == "fun*" => {
              let parameters = tail
                .get(0)
                .cloned()
                .ok_or(symbol!("error/expected-expr"))?
                .symbol_list()?;
              let body = tail.get(1).ok_or(symbol!("error/expected-expr"))?.clone();

              return Ok(Value::Fun(Fun {
                parameters,
                body: body.into(),
                env: EnvStruct::from(env),
              }));
            }
            Value::Symbol(ref name) if name == "def*" => {
              let name = tail
                .get(0)
                .cloned()
                .ok_or(symbol!("error/expected-expr"))?
                .name()?;
              let value = tail
                .get(1)
                .ok_or(symbol!("error/expected-expr"))?
                .clone()
                .eval(env)?;

              env.define(name.clone(), value);

              return Ok(Value::Symbol(name));
            }
            Value::Symbol(ref name) if name == "defmacro*" => {
              let name = tail
                .get(0)
                .cloned()
                .ok_or(symbol!("error/expected-expr"))?
                .name()?;
              let value = tail
                .get(1)
                .ok_or(symbol!("error/expected-expr"))?
                .clone()
                .eval(env)?;

              if let Value::Fun(fun) = value {
                env.define(name, Value::Macro(fun));
                return Ok(Value::Nil);
              } else {
                return Err(symbol!("error/macro-definition-to-non-function"));
              }
            }
            _ => match head.clone().eval(env)? {
              Value::Fun(fun) => {
                let arguments = tail
                  .iter()
                  .cloned()
                  .map(|e| e.eval(env))
                  .collect::<Result<_, _>>()?;
                return fun.call(arguments);
              }
              _ => break Err(symbol!("error/call-to-non-function")),
            },
          }
        }
        a => unreachable!("{a:?}"),
      }
    }
  }

  pub fn macroexpand(mut self, env: &Env) -> Result<Value, Value> {
    'macroexpansion: loop {
      match self {
        Value::List(list) => {
          if let Some((head, tail)) = list.split_first() {
            match head.clone().macroexpand(env)? {
              Value::Macro(mac) => {
                let arguments = tail
                  .iter()
                  .cloned()
                  .map(|e| e.macroexpand(env))
                  .collect::<Result<_, _>>()?;
                self = mac.call(arguments)?;
                continue 'macroexpansion;
              }
              _ => break Ok(Value::List(list)),
            }
          } else {
            break Ok(Value::List(list));
          }
        }
        Value::Symbol(ref name) => {
          if let Some(mac @ Value::Macro(_)) = env.fetch(name) {
            self = mac;
            continue 'macroexpansion;
          } else {
            break Ok(self);
          }
        }
        _ => break Ok(self),
      }
    }
  }

  pub fn eval(mut self, env: &Env) -> Result<Value, Value> {
    self = self.macroexpand(env)?;
    match self {
      Value::Symbol(name) => match env.fetch(&name) {
        Some(value) => Ok(value),
        None => Err(list![
          symbol!("error/undefined-symbol"),
          Value::Symbol(name)
        ]),
      },

      Value::List(ref list) if list.is_empty() => Ok(Value::Nil),
      Value::List(_) => self.apply(env),

      Value::Number(num) => Ok(Value::Number(num)),
      Value::String(str) => Ok(Value::String(str)),
      Value::Atom(atom) => Ok(Value::Atom(atom)),
      Value::Fun(fun) => Ok(Value::Fun(fun)),
      Value::Macro(mac) => Ok(Value::Macro(mac)),
      Value::Nil => Ok(Value::Nil),
    }
  }
}

fn main() -> miette::Result<()> {
  use clap::Parser;

  let cli = Cli::parse();
  let source = Source::try_from(cli.main)?;
  let exprs = parse_from_source(source)?;

  let ref env = Env::new(EnvStruct::default());

  for expr in exprs {
    if let Err(e) = Value::from(expr).eval(env) {
      eprintln!("{e}");
    }
  }

  Ok(())
}
