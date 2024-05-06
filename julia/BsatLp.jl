using COSMO, LinearAlgebra, SparseArrays

function parse_dimacs(filepath::String)
  # Open the file for reading
  open(filepath, "r") do file
      n_variables = 0
      n_clauses = 0
      clause_list = []

      # Process each line in the file
      for line in eachline(file)
          # Ignore comments and empty lines
          if startswith(line, "c") || isempty(line)
              continue
          elseif startswith(line, "p")
              # Extract problem definition line
              tokens = split(line)
              if length(tokens) != 4 || tokens[1] != "p" || tokens[2] != "cnf"
                  error("Invalid DIMACS header format")
              end
              n_variables = parse(Int, tokens[3])
              n_clauses = parse(Int, tokens[4])
          elseif startswith(line, "%")
            break
          else
              # Process a clause line
              tokens = split(line)
              parsed_clause = parse.(Int, tokens)
              if last(parsed_clause) != 0
                  error("Each clause line must end with a zero.")
              end
              push!(clause_list, parsed_clause[1:end-1])  # Exclude the ending zero
          end
      end

      # Verify that the number of clauses matches the header
      if length(clause_list) != n_clauses
          error("The number of clauses does not match the header.")
      end

      return n_variables, clause_list
  end
end

n_variables = 0
clauses = []
input_filepath = "bin/uf20-01.cnf"
try
  global n_variables, clauses
  n_variables, clauses = parse_dimacs(input_filepath)
catch e
  println("Error parsing DIMACS file: ", e)
end

println("Number of variables: ", n_variables)
println("Clauses: ", clauses)
setprecision(BigFloat, n_variables*2)

A_rows = Int[]
A_cols = Int[]
A_vals = BigFloat[]
l = Vector{BigFloat}(undef, length(clauses) + n_variables)
u = Vector{BigFloat}(undef, length(clauses) + n_variables)
P = spzeros(BigFloat, n_variables, n_variables)
q = Vector{BigFloat}(undef, n_variables)
asgs = falses(n_variables)

i_constr = 0

for a_var in 1:n_variables
  global i_constr
  i_constr += 1

  push!(A_rows, i_constr)
  push!(A_cols, a_var)
  push!(A_vals, BigFloat(1))
  l[i_constr] = -1
  u[i_constr] = 1
  q[i_constr] = BigFloat(2.0) ^ (-i_constr)
end

for clause in clauses
  global i_constr
  i_constr += 1
  for i_var in clause
    push!(A_rows, i_constr)
    push!(A_cols, abs(i_var))
    push!(A_vals, sign(i_var))
  end
  l[i_constr] = 2.0 - length(clause)
  u[i_constr] = Inf
end

A = sparse(A_rows, A_cols, A_vals)

model = COSMO.Model{BigFloat}()
constraint = COSMO.Constraint(A, zeros(BigFloat, n_variables+length(clauses)), COSMO.Box(l, u))
settings = COSMO.Settings{BigFloat}(verbose = true, max_iter=Int(1e10))
assemble!(model, P, q, constraint, settings = settings)

const eps = 1e-3
maybe_sat = true
tot_its = 0;
while maybe_sat
  global maybe_sat, tot_its
  tot_its += 1
  x_warm = Vector{BigFloat}(undef, n_variables)
  for i in 1:n_variables
    if asgs[i]
      x_warm[i] = 1
    else
      x_warm[i] = -1
    end
  end
  COSMO.warm_start_primal!(model, x_warm)
  lp_res = COSMO.optimize!(model);
  if lp_res.status != :Solved && lp_res.status != :SolvedSuboptimal
    maybe_sat = false
    println("UNSATISFIABLE: ", lp_res.status)
    break
  end
  x_sol = lp_res.x
  def_vars = []
  undef_vars = []
  for i in 1:n_variables
    val = x_sol[i]
    if abs(val) < 1-eps
      push!(undef_vars, i)
    else
      push!(def_vars, i)
      if val > -0
        asgs[i] = true
      else
        asgs[i] = false
      end
    end
  end
  if length(undef_vars) == 0
    println("SATISFIED")
    break
  end
  sort!(undef_vars, lt=(a, b) -> abs(x_sol[a]) < abs(x_sol[b]))
  sort!(def_vars, lt=(a, b) -> abs(x_sol[a]) < abs(x_sol[b]))
  var_ord = [undef_vars; def_vars]
  for i in 1:n_variables
    q[var_ord[i]] = BigFloat(2.0) ^ (-i)
  end
  update!(model, q = q)
end

println("Number of top-level iterations: ", tot_its)
