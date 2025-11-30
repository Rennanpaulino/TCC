package com.seutcc.app;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.adapters.TremAdapter;
import com.seutcc.app.models.Trem;
import com.seutcc.app.network.RetrofitClient;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    private RecyclerView recycler;
    private TremAdapter adapter;
    private Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Botão para ir aos gráficos
        Button btnRelatorios = findViewById(R.id.btnRelatorios);
        btnRelatorios.setOnClickListener(v -> startActivity(new Intent(this, RelatoriosActivity.class)));

        // Configura Lista
        recycler = findViewById(R.id.recyclerTrens);
        recycler.setLayoutManager(new LinearLayoutManager(this));

        // Ao clicar no trem, vai para Detalhes
        adapter = new TremAdapter(trem -> {
            Intent intent = new Intent(MainActivity.this, DetalhesTremActivity.class);
            intent.putExtra("ID_TREM", trem.getId()); // Passa o ID pra outra tela
            startActivity(intent);
        });
        recycler.setAdapter(adapter);

        // Inicia Loop de Atualização
        atualizarDados();
    }

    private void atualizarDados() {
        RetrofitClient.getIoTService().getTrens().enqueue(new Callback<List<Trem>>() {
            @Override
            public void onResponse(Call<List<Trem>> call, Response<List<Trem>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    adapter.atualizarLista(response.body());
                }
            }
            @Override
            public void onFailure(Call<List<Trem>> call, Throwable t) {}
        });

        // Repete a cada 3 segundos
        handler.postDelayed(this::atualizarDados, 3000);
    }
}