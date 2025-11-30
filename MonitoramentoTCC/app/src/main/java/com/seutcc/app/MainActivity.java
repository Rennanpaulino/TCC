package com.seutcc.app;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button; // <--- NÃO ESQUEÇA DESTE IMPORT
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.models.Linha;
import com.seutcc.app.network.RetrofitClient;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // --- 1. CONFIGURA O BOTÃO DO DASHBOARD ---
        // Aqui nós "achamos" o botão que criamos no XML pelo ID
        Button btnDashboard = findViewById(R.id.btnDashboard);

        // Aqui definimos o que acontece ao clicar
        btnDashboard.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, RelatoriosActivity.class);
            startActivity(intent);
        });

        // --- 2. CONFIGURA A LISTA DE LINHAS (RECYCLER) ---
        RecyclerView rv = findViewById(R.id.recyclerLinhas);
        rv.setLayoutManager(new LinearLayoutManager(this));

        // Busca dados do Backend
        buscarLinhas(rv);
    }

    private void buscarLinhas(RecyclerView rv) {
        RetrofitClient.getIoTService().getLinhas().enqueue(new Callback<List<Linha>>() {
            @Override
            public void onResponse(Call<List<Linha>> call, Response<List<Linha>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    // Configura o Adapter com a lista recebida
                    rv.setAdapter(new LinhaAdapter(response.body()));
                } else {
                    Toast.makeText(MainActivity.this, "Erro ao carregar linhas", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<List<Linha>> call, Throwable t) {
                Toast.makeText(MainActivity.this, "Sem conexão com o servidor", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // --- ADAPTER INTERNO (Lógica visual da lista) ---
    class LinhaAdapter extends RecyclerView.Adapter<LinhaAdapter.Holder> {
        List<Linha> dados;

        public LinhaAdapter(List<Linha> dados) { this.dados = dados; }

        @NonNull @Override
        public Holder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext()).inflate(android.R.layout.simple_list_item_1, parent, false);
            return new Holder(v);
        }

        @Override
        public void onBindViewHolder(@NonNull Holder holder, int position) {
            Linha l = dados.get(position);
            holder.text.setText(l.getNome());

            // Tenta pintar o texto com a cor da linha (ex: #00A88E)
            try {
                holder.text.setTextColor(Color.parseColor(l.getCorHex())); // No model novo chamamos getCorHex
                holder.text.setTextSize(18);
                holder.text.setTypeface(null, android.graphics.Typeface.BOLD);
            } catch (Exception e) {
                holder.text.setTextColor(Color.BLACK);
            }

            // Clique na Linha -> Vai para Estações
            holder.itemView.setOnClickListener(v -> {
                Intent i = new Intent(MainActivity.this, EstacoesActivity.class);
                i.putExtra("LINHA_ID", l.getId());
                i.putExtra("LINHA_NOME", l.getNome());
                startActivity(i);
            });
        }

        @Override public int getItemCount() { return dados.size(); }

        class Holder extends RecyclerView.ViewHolder {
            TextView text;
            public Holder(@NonNull View itemView) {
                super(itemView);
                text = itemView.findViewById(android.R.id.text1);
            }
        }
    }
}