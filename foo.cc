

class Parameters {

  std::map<std::string, FunctionParameter> parameters;

  const static std::map<std::string, std::tuple<void *, std::typeinfo, std::typeinfo>>
      function_map;

  void set_relaxation_type(Parameters::Relaxation relaxation, Parameters::Relaxation relaxation)
  {
    // ...
  }


  template<typename T>
  void add_parameter(std::string name, T(*)(T), const T &)
  {
  }


  template<typename T>
  void add_parameter(std::string name, T(*)(T, T), const T &, const T&)
  {
  }


  add_parameter("blablabla", &HYPRE_...,


  template<typename T>
  void set_parameter(std::string name, const T &value)
  {
    AssertThrow(function_map.find(...), ...);

    AssertThrow(function_map[name].second == typeid(value), ...);

    AssertThrow( ... == typeid(...));

    parameters[name] = FunctionParameter(
        Hypre_Chooser::Solver,
        static_cast<T (*)(T)>(function_map[name].first), value);
  }


  template<typename T1, typename T2>
  void set_parameter(std::string name, const T1 &val1, const T2 &val2)
  {
    typeid(make_pair(val1, val2))
  }


  {
    std::vector<FunctionParameter *> foo;
    for(auto it : function_map)
      foo.push_back(&it);

    foo.data();
    foo.length();
  }

};

Parameters::function_map = {
    {"foo1", {&bar_1, typeid(&bar_1), typeid(int)}},
    {"foo2", {&bar_2, typeid(double)}},
};

...
