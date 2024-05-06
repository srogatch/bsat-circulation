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

var_order = collect(1:n_variables)
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
  q[i_constr] = 2.0^(-i_constr)
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
@show q
@show l
@show u

model = COSMO.Model{BigFloat}()
constraint = COSMO.Constraint(A, zeros(BigFloat, n_variables+length(clauses)), COSMO.Box(l, u))
settings = COSMO.Settings{BigFloat}(verbose = true)
assemble!(model, P, q, constraint, settings = settings)
result = COSMO.optimize!(model);
# println(result)
